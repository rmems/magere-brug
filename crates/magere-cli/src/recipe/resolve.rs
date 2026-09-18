use super::{RECIPE_SCHEMA, Recipe};
use crate::manifest::Manifest;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A reference is treated as a filesystem path when it names a JSON file;
/// anything else is a registry id resolved elsewhere.
pub(super) fn reference_is_manifest_path(reference: &str) -> bool {
    reference.ends_with(".json")
}

/// True when a reference contains a `..` path segment.
///
/// Checked as a segment rather than a substring so that legitimate names
/// containing dots (`model..v2.json`) are not rejected.
pub(super) fn reference_escapes_upward(reference: &str) -> bool {
    Path::new(reference)
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
}

/// Marks the boundary the reference search must not cross.
///
/// `.git` is a directory in a normal clone and a file in a linked worktree, so
/// `exists()` covers both; `Cargo.lock` is the fallback for an exported tree
/// with no VCS metadata.
fn is_repo_root(dir: &Path) -> bool {
    dir.join(".git").exists() || dir.join("Cargo.lock").is_file()
}

pub(super) fn validate_against_schema(instance: &Value) -> Result<(), String> {
    let schema: Value = serde_json::from_str(RECIPE_SCHEMA)
        .map_err(|e| format!("embedded recipe schema is not valid JSON: {e}"))?;

    let validator = jsonschema::draft7::new(&schema)
        .map_err(|e| format!("embedded recipe schema failed to compile: {e}"))?;

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| {
            let location = error.instance_path().to_string();
            if location.is_empty() {
                format!("<root>: {error}")
            } else {
                format!("{location}: {error}")
            }
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "recipe schema validation failed:\n  - {}",
            errors.join("\n  - ")
        ))
    }
}

impl Recipe {
    /// Input references paired with the file each one actually resolved to.
    ///
    /// Only path-shaped references that resolve are reported; registry ids and
    /// unresolvable references are skipped (validation reports those as errors).
    pub fn resolved_references(&self) -> Vec<(String, PathBuf)> {
        let mut out = Vec::new();
        for (label, reference) in [
            ("inputs.source_manifest", self.source_manifest_ref()),
            ("inputs.goz1_ref", self.goz1_ref()),
        ] {
            if let Some(reference) = reference
                && reference_is_manifest_path(reference)
                && !reference_escapes_upward(reference)
                && let Some(resolved) = self.resolve_reference(reference)
            {
                out.push((label.to_string(), resolved));
            }
        }
        out
    }

    pub(super) fn load_referenced_manifest(
        &self,
        reference: &str,
        label: &str,
    ) -> Result<Option<Manifest>, String> {
        if !reference_is_manifest_path(reference) {
            return Err(format!(
                "{label} '{reference}' must name a manifest path ending in '.json'; \
                 registry-id references are not resolvable yet (see issues #19/#8)"
            ));
        }

        if reference_escapes_upward(reference) {
            return Err(format!(
                "{label} '{reference}' must not contain '..' segments; recipe inputs are \
                 resolved within the repository that contains the recipe"
            ));
        }

        let resolved = self.resolve_reference(reference).ok_or_else(|| {
            format!(
                "{label} '{reference}' could not be resolved to a file on disk \
                 (searched the recipe's directory up to the repository root that \
                 contains it, and no further)"
            )
        })?;

        let manifest = Manifest::from_file(&resolved).map_err(|e| {
            format!(
                "{label} '{}' is not a parseable manifest: {e}",
                resolved.display()
            )
        })?;

        manifest.validate().map_err(|e| {
            format!(
                "{label} '{}' is not a valid manifest: {e}",
                resolved.display()
            )
        })?;

        Ok(Some(manifest))
    }

    /// Resolve an input reference that must already exist on disk.
    ///
    /// The search is **bounded before it begins**, so it can never reach outside the
    /// tree the recipe lives in.
    pub(super) fn resolve_reference(&self, reference: &str) -> Option<PathBuf> {
        let candidate = Path::new(reference);

        if candidate.is_absolute() {
            return candidate.is_file().then(|| candidate.to_path_buf());
        }

        let Some(dir) = self.source_path.as_deref().and_then(Path::parent) else {
            return candidate.is_file().then(|| candidate.to_path_buf());
        };

        for ancestor in self.search_roots(dir) {
            let joined = ancestor.join(candidate);
            if joined.is_file() {
                return Some(joined);
            }
        }

        None
    }

    /// Directories a relative reference may be resolved against, nearest first.
    fn search_roots<'a>(&self, dir: &'a Path) -> Vec<&'a Path> {
        match dir.ancestors().find(|ancestor| is_repo_root(ancestor)) {
            Some(root) => dir
                .ancestors()
                .take_while(|ancestor| *ancestor != root)
                .chain(std::iter::once(root))
                .collect(),
            None => vec![dir],
        }
    }
}
