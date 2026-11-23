//! Output layout, path collision checks, and source-manifest resolution.

use crate::manifest::Manifest;
use crate::recipe::{Recipe, is_filename_safe_id};
use crate::registry::{ArtifactRegistry, RegistryEntry};
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

/// Derived pack and manifest destinations for one run.
#[derive(Debug, Clone)]
pub(super) struct OutputLayout {
    pub output_dir: PathBuf,
    pub manifest_id: String,
    pub pack_path: PathBuf,
    pub manifest_path: PathBuf,
}

pub(super) fn resolve_output_layout(
    recipe: &Recipe,
    source: &Manifest,
    output_dir_override: Option<&Path>,
) -> Result<OutputLayout, String> {
    let (output_dir, dir_label) = match output_dir_override {
        Some(dir) => (dir.to_path_buf(), "--output-dir"),
        None => (
            recipe
                .outputs
                .as_ref()
                .and_then(|o| o.output_dir.as_deref())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    format!(
                        "recipe '{}' has no outputs.output_dir; pass --output-dir",
                        recipe.recipe_id
                    )
                })?,
            "outputs.output_dir",
        ),
    };
    reject_parent_dir_components(&output_dir, dir_label)?;

    let manifest_id = recipe
        .outputs
        .as_ref()
        .and_then(|o| o.manifest_id.clone())
        .unwrap_or_else(|| format!("{}-goz1", source.metadata.manifest_id));
    let manifest_id = filename_stem(&manifest_id)?.to_string();
    let pack_path = output_dir.join(format!("{}.goz1", manifest_id));
    let manifest_path = output_dir.join(format!("{}.manifest.json", manifest_id));
    Ok(OutputLayout {
        output_dir,
        manifest_id,
        pack_path,
        manifest_path,
    })
}

/// Load the source manifest from a filesystem path or a registry slug / manifest_id.
pub(super) fn resolve_source_manifest(
    recipe: &Recipe,
    registry: &ArtifactRegistry,
) -> Result<(PathBuf, Manifest), String> {
    let raw = recipe
        .inputs
        .as_ref()
        .and_then(|i| i.source_manifest.as_deref())
        .ok_or_else(|| {
            format!(
                "recipe '{}' is missing inputs.source_manifest",
                recipe.recipe_id
            )
        })?;
    let path = resolve_source_ref(raw, recipe, registry)?;
    let source = Manifest::from_file(&path)
        .map_err(|e| format!("Failed to load source manifest '{}': {}", path.display(), e))?;
    source
        .validate()
        .map_err(|e| format!("source manifest '{}' is invalid: {}", path.display(), e))?;
    Ok((path, source))
}

fn resolve_source_ref(
    raw: &str,
    recipe: &Recipe,
    registry: &ArtifactRegistry,
) -> Result<PathBuf, String> {
    if let Some(resolved) = resolve_manifest_path_reference(raw, recipe)? {
        return Ok(resolved);
    }
    // A bare `*.json` name may be a registry id; inside a repository the cwd
    // exists() probe must not reopen the checkout boundary the bounded search
    // just refused.
    let skip_fs_probe = recipe.inside_repository() && is_bare_manifest_name(raw);
    resolve_path_or_registry_reference(raw, registry, skip_fs_probe)
}

/// A bare `*.json` filename — no directory component — that may name a
/// registry slug or manifest_id rather than a filesystem path.
fn is_bare_manifest_name(raw: &str) -> bool {
    !raw.contains('/') && !raw.contains('\\') && raw.ends_with(".json")
}

/// Resolve a `*.json`-looking reference against the recipe's search rules.
///
/// Returns `Ok(None)` when `raw` is not manifest-path shaped, or when it is
/// shaped like one but none of the file rules apply — the caller then falls
/// through to plain-path and registry-id resolution.
fn resolve_manifest_path_reference(raw: &str, recipe: &Recipe) -> Result<Option<PathBuf>, String> {
    use crate::recipe::resolve::{reference_escapes_upward, reference_is_manifest_path};

    if !reference_is_manifest_path(raw) {
        return Ok(None);
    }
    if let Some(resolved) = recipe.resolve_reference(raw) {
        return Ok(Some(resolved));
    }
    let as_manifest_path = Path::new(raw);
    if as_manifest_path.is_absolute() {
        return canonicalize_existing(as_manifest_path, "source manifest").map(Some);
    }
    if reference_escapes_upward(raw) {
        return Err(format!(
            "inputs.source_manifest '{}' must not contain '..' segments; recipe inputs are \
             resolved within the repository that contains the recipe",
            raw
        ));
    }
    resolve_relative_manifest_reference(raw, as_manifest_path, recipe)
}

/// Handle a relative manifest-path reference that the bounded recipe search
/// did not resolve: the outside-repository cwd fallback, then the
/// missing-path vs registry-id decision.
fn resolve_relative_manifest_reference(
    raw: &str,
    as_manifest_path: &Path,
    recipe: &Recipe,
) -> Result<Option<PathBuf>, String> {
    // A relative manifest path may also live relative to the caller's working
    // directory — but only when the recipe sits outside any repository. Inside
    // a checkout the bounded search above is the whole contract; a cwd
    // fallback could silently load a same-named file from a different
    // checkout.
    if !recipe.inside_repository() && as_manifest_path.exists() {
        return Ok(Some(as_manifest_path.to_path_buf()));
    }
    // A bare `*.json` filename may still be a registry id — manifest_ids can
    // end in `.json` — so only spellings with a directory component are
    // unambiguously missing filesystem paths; the rest fall through to
    // registry lookup.
    if raw.contains('/') || raw.contains('\\') {
        return Err(format!(
            "Failed to load source manifest '{}': file not found",
            raw
        ));
    }
    Ok(None)
}

/// Resolve `raw` as an existing filesystem path, else as a registry slug or
/// manifest_id carrying a stored `manifest_path`.
///
/// `skip_fs_probe` suppresses the cwd-relative `exists()` check for bare
/// manifest names inside a repository, where the bounded search is the whole
/// contract.
fn resolve_path_or_registry_reference(
    raw: &str,
    registry: &ArtifactRegistry,
    skip_fs_probe: bool,
) -> Result<PathBuf, String> {
    let as_path = Path::new(raw);
    if !skip_fs_probe && as_path.exists() {
        return Ok(as_path.to_path_buf());
    }
    let entry = registry
        .lookup(raw)
        .or_else(|| registry.lookup_by_manifest_id(raw))
        .ok_or_else(|| {
            format!(
                "inputs.source_manifest '{}' is not a readable path and is not a slug or \
                 manifest_id in the selected registry",
                raw
            )
        })?;
    stored_entry_manifest_path(raw, entry)
}

fn stored_entry_manifest_path(raw: &str, entry: &RegistryEntry) -> Result<PathBuf, String> {
    match &entry.manifest_path {
        Some(path) if path.is_file() => Ok(path.clone()),
        Some(path) => Err(format!(
            "registry id '{}' points at missing manifest '{}'",
            raw,
            path.display()
        )),
        None => Err(format!(
            "registry id '{}' is registered but has no stored manifest_path; re-register \
             the source with `magere register`",
            raw
        )),
    }
}

/// One filename component, using the same rule as other recipe output ids.
fn filename_stem(manifest_id: &str) -> Result<&str, String> {
    let trimmed = manifest_id.trim();
    if is_filename_safe_id(trimmed) {
        return Ok(trimmed);
    }
    Err(format!(
        "outputs.manifest_id '{manifest_id}' must be a single filename with no path separators or '..' segments"
    ))
}

/// Refuse when a derived output path would overwrite a recipe input or the registry file.
///
/// `registry_paths` carries every spelling of the registry destination — the
/// user-spelled path and its resolved target. Each is protected as the
/// directory *entry* it names: a symlink sitting at an output path is itself
/// replaced by the atomic write, so protecting only the link's target would
/// leave the spelled path holding artifact bytes.
pub(super) fn reject_aliased_outputs(
    recipe_path: &Path,
    source_manifest_path: &Path,
    dissect_manifest_path: &Path,
    registry_paths: &[&Path],
    pack_path: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    let mut protected = Vec::with_capacity(6);
    for input in [recipe_path, source_manifest_path, dissect_manifest_path] {
        protected.push(canonicalize_existing(input, "input")?);
    }
    for registry_path in registry_paths {
        protected.push(resolve_entry(registry_path)?);
        // Intermediate links in a symlink chain are entries too: an output
        // replacing one would leave the registry resolving to artifact
        // bytes even though the spelled path and final target are protected.
        for link in crate::registry_io::symlink_chain_entries(registry_path) {
            protected.push(resolve_entry(&link)?);
        }
    }
    for output in [pack_path, manifest_path] {
        let resolved = resolve_planned(output)?;
        if let Some(hit) = protected.iter().find(|path| paths_alias(path, &resolved)) {
            return Err(format!(
                "refusing to run: output '{}' would overwrite recipe input '{}'",
                output.display(),
                hit.display()
            ));
        }
    }
    Ok(())
}

fn canonicalize_existing(path: &Path, kind: &str) -> Result<PathBuf, String> {
    std::fs::canonicalize(path)
        .map_err(|e| format!("Failed to resolve {kind} '{}': {}", path.display(), e))
}

pub(super) fn resolve_planned_path(path: &Path) -> Result<PathBuf, String> {
    resolve_planned(path)
}

fn resolve_planned(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return canonicalize_existing(path, "path");
    }
    resolve_entry(path)
}

/// Resolve `path` as the directory entry it names without following a
/// final-component symlink: the parent side is canonicalized and the file
/// name stays literal, so a symlink and the file it points at are distinct
/// entries. `resolve_planned` cannot express this — it canonicalizes existing
/// paths, which follows the link to its target.
fn resolve_entry(path: &Path) -> Result<PathBuf, String> {
    let (parent, file_name) = split_planned(path)?;
    resolve_under_parent(parent, file_name).map(|resolved| normalize_lexically(&resolved))
}

fn split_planned(path: &Path) -> Result<(&Path, &OsStr), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path '{}' has no parent", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path '{}' has no file name", path.display()))?;
    Ok((parent, file_name))
}

fn resolve_under_parent(parent: &Path, file_name: &OsStr) -> Result<PathBuf, String> {
    if parent.as_os_str().is_empty() {
        return current_dir().map(|cwd| cwd.join(file_name));
    }

    let (ancestor, suffix) = nearest_existing_ancestor(parent, file_name);
    if ancestor.as_os_str().is_empty() {
        return current_dir().map(|cwd| cwd.join(parent).join(file_name));
    }

    let canonical_parent = canonicalize_existing(&ancestor, "output dir")?;
    Ok(canonical_parent.join(suffix))
}

fn current_dir() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|e| format!("Failed to resolve current directory: {e}"))
}

/// Walk `parent` up to the nearest existing ancestor, accumulating the missing
/// tail into `suffix`. An empty `ancestor` means no component exists. Missing
/// `..` components are kept in `suffix` — `file_name` drops them, but they
/// still matter once the missing directories exist.
fn nearest_existing_ancestor(parent: &Path, file_name: &OsStr) -> (PathBuf, PathBuf) {
    let mut suffix = PathBuf::from(file_name);
    let mut cursor = parent;
    while !cursor.as_os_str().is_empty() && !cursor.exists() {
        match cursor.file_name() {
            Some(name) => suffix = PathBuf::from(name).join(&suffix),
            None if cursor.ends_with("..") => suffix = PathBuf::from("..").join(&suffix),
            None => {}
        }
        cursor = cursor.parent().unwrap_or(Path::new(""));
    }
    (cursor.to_path_buf(), suffix)
}

/// Refuse `..` components in output paths. The lexical fold in
/// [`resolve_planned`] cannot model the filesystem when `..` follows a
/// symlink — a missing directory can hide one until it is created — so the
/// checked destination could differ from where the writer lands.
pub(super) fn reject_parent_dir_components(path: &Path, label: &str) -> Result<(), String> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "{label} '{}' must not contain '..' components; spell the output path \
             without parent traversal",
            path.display()
        ));
    }
    Ok(())
}

/// Fold `.`/`..` lexically so a planned path whose missing tail escapes an
/// ancestor (`packs/missing/../pack.goz1`) compares equal to the path the
/// writer will actually create. `..` that climbs past the root is dropped, as
/// the filesystem would; `..` under a relative root is preserved.
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir if normalized.file_name().is_some() => {
                normalized.pop();
            }
            Component::ParentDir if !normalized.has_root() => normalized.push(".."),
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn paths_alias(left: &Path, right: &Path) -> bool {
    left == right || same_file::is_same_file(left, right).unwrap_or(false)
}

/// Refuse when a derived output path would overwrite another registry entry's files.
///
/// Slug and manifest-id ownership live on [`ArtifactRegistry::admit_pack`]. The
/// same-slug entry is skipped only when the run will register — an admitted
/// replay replaces its own outputs — because an unregistered run does not own
/// those files and must not overwrite them either.
pub(super) fn reject_output_aliases(
    registry: &ArtifactRegistry,
    slug: &str,
    pack_path: &Path,
    manifest_path: &Path,
    will_register: bool,
) -> Result<(), String> {
    for entry in registry.models.values() {
        if will_register && entry.slug == slug {
            continue;
        }
        if entry_aliases_outputs(entry, pack_path, manifest_path)? {
            return Err(format!(
                "output path '{}' collides with registry slug '{}' (manifest '{}'). \
                 Nothing was written.",
                pack_path.display(),
                entry.slug,
                entry
                    .manifest_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| entry.manifest_id.clone())
            ));
        }
    }
    Ok(())
}

fn entry_aliases_outputs(
    entry: &RegistryEntry,
    pack_path: &Path,
    manifest_path: &Path,
) -> Result<bool, String> {
    let Some(stored) = &entry.manifest_path else {
        return Ok(false);
    };
    let ours = [resolve_planned(pack_path)?, resolve_planned(manifest_path)?];
    let mut theirs = vec![resolve_planned(stored)?];
    if let Some(pack) = sibling_goz1(stored) {
        theirs.push(resolve_planned(&pack)?);
    }
    Ok(ours
        .iter()
        .any(|output| theirs.iter().any(|existing| paths_alias(output, existing))))
}

fn sibling_goz1(manifest: &Path) -> Option<PathBuf> {
    let name = manifest.file_name()?.to_string_lossy();
    let stem = name.strip_suffix(".manifest.json")?;
    Some(manifest.with_file_name(format!("{stem}.goz1")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::recipe::load_pack_recipe;
    use crate::registry::ArtifactRegistry;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const SOURCE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../manifests/examples/grok-1-future-plan.json"
    );
    const DISSECT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/recipes/fixtures/grok-mini-dissect.json"
    );

    #[test]
    fn resolves_source_manifest_beside_the_recipe_file() {
        let dir = TempDir::new().unwrap();
        let manifests = dir.path().join("manifests");
        fs::create_dir_all(&manifests).unwrap();
        fs::copy(PathBuf::from(SOURCE), manifests.join("beside.json")).unwrap();
        let recipes = dir.path().join("configs/recipes");
        fs::create_dir_all(&recipes).unwrap();
        let recipe_path = recipes.join("pack.json");
        let recipe_body = json!({
            "recipe_id": "beside-pack",
            "type": "ternary_pack",
            "inputs": { "source_manifest": "../../manifests/beside.json" },
            "outputs": {
                "generated_format": "goz1",
                "manifest_id": "test-pack-v1",
                "output_dir": "packs"
            },
            "pack": {
                "dissect_manifest": DISSECT,
                "input_format": "safetensors"
            }
        });
        fs::write(
            &recipe_path,
            serde_json::to_string_pretty(&recipe_body).unwrap(),
        )
        .unwrap();
        let recipe = load_pack_recipe(&recipe_path).unwrap();
        let registry = ArtifactRegistry::new();
        let (resolved, _) = resolve_source_manifest(&recipe, &registry).unwrap();
        assert!(resolved.ends_with("manifests/beside.json"));
        assert!(resolved.is_file());
    }

    #[test]
    fn a_relative_manifest_path_outside_the_repo_resolves_from_cwd() {
        // The recipe lives outside any repository, so the recipe-relative
        // search finds nothing; the reference still resolves from the working
        // directory (the crate root under cargo test).
        let (_cwd_file, name) = cwd_manifest_fixture("cwd-fallback");

        let dir = TempDir::new().unwrap();
        let recipe_body = json!({
            "recipe_id": "cwd-pack",
            "type": "ternary_pack",
            "inputs": { "source_manifest": name },
            "outputs": {
                "generated_format": "goz1",
                "manifest_id": "test-pack-v1",
                "output_dir": "packs"
            },
            "pack": {
                "dissect_manifest": DISSECT,
                "input_format": "safetensors"
            }
        });
        let recipe_path = dir.path().join("pack.json");
        fs::write(
            &recipe_path,
            serde_json::to_string_pretty(&recipe_body).unwrap(),
        )
        .unwrap();
        let recipe = load_pack_recipe(&recipe_path).unwrap();
        let registry = ArtifactRegistry::new();
        let resolved = resolve_source_manifest(&recipe, &registry);
        let (resolved, _) = resolved.expect("cwd-relative manifest must resolve");
        assert!(resolved.ends_with(&name));
    }

    #[test]
    fn an_in_repo_recipe_does_not_fall_back_to_cwd() {
        // The recipe lives inside its own repository boundary, so a manifest
        // that resolves nowhere inside it must not silently load a same-named
        // file from the process working directory (a different checkout).
        let repo = TempDir::new().unwrap();
        fs::create_dir(repo.path().join(".git")).unwrap();
        let (_cwd_file, name) = cwd_manifest_fixture("cwd-leak");

        let recipe_body = json!({
            "recipe_id": "in-repo-pack",
            "type": "ternary_pack",
            "inputs": { "source_manifest": name },
            "outputs": {
                "generated_format": "goz1",
                "manifest_id": "test-pack-v1",
                "output_dir": "packs"
            },
            "pack": {
                "dissect_manifest": DISSECT,
                "input_format": "safetensors"
            }
        });
        let recipe_path = repo.path().join("pack.json");
        fs::write(
            &recipe_path,
            serde_json::to_string_pretty(&recipe_body).unwrap(),
        )
        .unwrap();
        let recipe = load_pack_recipe(&recipe_path).unwrap();
        let registry = ArtifactRegistry::new();
        let resolved = resolve_source_manifest(&recipe, &registry);
        let err = resolved.expect_err("an in-repo recipe must not resolve manifests from cwd");
        assert!(
            err.contains("file not found") || err.contains("not a readable path"),
            "{err}"
        );
    }

    #[test]
    fn planned_paths_under_new_subdirs_share_a_canonical_root() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("pack-root");
        fs::create_dir_all(&base).unwrap();
        // `new` never exists, so its `..` can only fold lexically after the
        // ancestor walk — the spellings agree only if `..` survives the walk
        // and normalizes onto the path the writer creates.
        let expected = std::fs::canonicalize(&base).unwrap().join("out/a.goz1");
        assert_eq!(
            resolve_planned_path(&base.join("out/a.goz1")).unwrap(),
            expected
        );
        assert_eq!(
            resolve_planned_path(&base.join("new/../out/a.goz1")).unwrap(),
            expected,
            "a `..` over a missing component must resolve to the same canonical path"
        );
        assert_eq!(
            resolve_planned_path(&base.join("new/../../a.goz1")).unwrap(),
            std::fs::canonicalize(base.parent().unwrap())
                .unwrap()
                .join("a.goz1"),
            "`..` escaping the ancestor must still resolve where the writer lands"
        );
    }

    fn registry_with_manifest_id(manifest_id: &str) -> ArtifactRegistry {
        let registry_json = json!({
            "version": 1,
            "models": {
                "baseline_source": {
                    "slug": "baseline_source",
                    "manifest_id": manifest_id,
                    "family": "grok",
                    "status": "registered",
                    "registered_at": "2026-08-01T00:00:00Z",
                    "manifest_path": SOURCE,
                    "producer": "register"
                }
            }
        });
        ArtifactRegistry::from_json(&registry_json.to_string()).unwrap()
    }

    /// An exclusively-created `*.json` manifest copy in the process working
    /// directory — the cwd-fallback tests need a real file there. The name is
    /// deterministic (pid + counter, no random characters that could spell
    /// the schema's banned substrings), creation is exclusive so a stale file
    /// cannot be overwritten, and the guard deletes it on drop.
    fn cwd_manifest_fixture(stem: &str) -> (tempfile::NamedTempFile, String) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let cwd = std::env::current_dir().unwrap();
        for _ in 0..8 {
            let base = format!(
                "{stem}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            match tempfile::Builder::new()
                .prefix(&base)
                .suffix(".json")
                .rand_bytes(0)
                .tempfile_in(&cwd)
            {
                Ok(file) => {
                    fs::copy(PathBuf::from(SOURCE), file.path()).unwrap();
                    return (file, format!("{base}.json"));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("failed to create a cwd fixture: {error}"),
            }
        }
        panic!("could not create a uniquely named cwd fixture");
    }

    fn pack_recipe_for(dir: &Path, source_manifest: &str) -> Recipe {
        let recipe_body = json!({
            "recipe_id": "registry-json-id-pack",
            "type": "ternary_pack",
            "inputs": { "source_manifest": source_manifest },
            "outputs": {
                "generated_format": "goz1",
                "manifest_id": "test-pack-v1",
                // A relative dir keeps the random tempdir name out of the
                // recipe: the schema refuses paths containing "awq"/"gptq",
                // which a random suffix can spell by chance.
                "output_dir": "packs"
            },
            "pack": {
                "dissect_manifest": DISSECT,
                "input_format": "safetensors"
            }
        });
        let recipe_path = dir.join("pack.json");
        fs::write(
            &recipe_path,
            serde_json::to_string_pretty(&recipe_body).unwrap(),
        )
        .unwrap();
        load_pack_recipe(&recipe_path).unwrap()
    }

    #[test]
    fn a_registered_manifest_id_ending_in_json_resolves_via_the_registry() {
        // manifest_ids may end in `.json`; a bare `*.json` reference that is
        // not a file must still reach registry lookup.
        let dir = TempDir::new().unwrap();
        let recipe = pack_recipe_for(dir.path(), "baseline.json");
        let registry = registry_with_manifest_id("baseline.json");
        let (resolved, manifest) = resolve_source_manifest(&recipe, &registry)
            .expect("a registered manifest_id ending in .json must resolve");
        assert_eq!(resolved, PathBuf::from(SOURCE));
        assert_eq!(manifest.metadata.manifest_id, "grok-1-future-plan-v1");
    }

    #[test]
    fn an_in_repo_bare_json_reference_still_reaches_the_registry() {
        // Inside a checkout the cwd exists() probe stays off for bare `*.json`
        // names — but a registered manifest_id must still resolve, and it must
        // win over a same-named file in the working directory.
        let repo = TempDir::new().unwrap();
        fs::create_dir(repo.path().join(".git")).unwrap();
        let (_cwd_file, name) = cwd_manifest_fixture("baseline");

        let recipe = pack_recipe_for(repo.path(), &name);
        let registry = registry_with_manifest_id(&name);
        let resolved = resolve_source_manifest(&recipe, &registry);
        let (resolved, _) = resolved.expect("the registry entry must resolve the bare name");
        assert_eq!(
            resolved,
            PathBuf::from(SOURCE),
            "the registered manifest_path wins over the cwd file"
        );
    }
}
