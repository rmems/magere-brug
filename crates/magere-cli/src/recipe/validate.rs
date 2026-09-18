use super::{
    EXPORTABLE_SOURCE_FORMATS, PACKABLE_SOURCE_FORMATS, Recipe, RecipeOutputs, RecipeType,
    SUPPORTED_GOZ1_VERSION, VALID_GENERATED_FORMATS, VALID_SOURCE_FORMATS,
};
use crate::manifest::Manifest;

impl Recipe {
    pub(super) fn validate_semantics(&self) -> Result<(), String> {
        self.require_inputs_for_type()?;
        self.reject_register_calibration()?;
        self.reject_unconsumed_saaq_fields()?;
        self.validate_declared_source_format()?;
        self.validate_outputs()?;
        self.validate_references()?;
        self.validate_saaq_runner_agreement()
    }

    fn require_inputs_for_type(&self) -> Result<(), String> {
        match self.recipe_type {
            RecipeType::Register
            | RecipeType::Goz1Pack
            | RecipeType::TernaryPack
            | RecipeType::GgufExport => {
                if self.source_manifest_ref().is_none() {
                    return Err(format!(
                        "recipe type '{}' requires inputs.source_manifest",
                        self.recipe_type
                    ));
                }
            }
            RecipeType::Saaq => self.require_saaq_inputs()?,
        }
        Ok(())
    }

    fn require_saaq_inputs(&self) -> Result<(), String> {
        if self.source_manifest_ref().is_none() && self.goz1_ref().is_none() {
            return Err(
                "recipe type 'saaq' requires inputs.source_manifest or inputs.goz1_ref".to_string(),
            );
        }
        match self.outputs.as_ref() {
            None => Err("recipe type 'saaq' requires outputs".to_string()),
            Some(outputs) if outputs.output_dir.is_none() => {
                Err("recipe type 'saaq' requires outputs.output_dir to place the run".to_string())
            }
            Some(_) => Ok(()),
        }
    }

    fn reject_register_calibration(&self) -> Result<(), String> {
        if self.recipe_type == RecipeType::Register && self.calibration.is_some() {
            return Err(
                "calibration is only meaningful for the ternary/goz1 pack and saaq paths; \
                 a register recipe must not carry it"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn reject_unconsumed_saaq_fields(&self) -> Result<(), String> {
        if self.recipe_type != RecipeType::Saaq {
            return Ok(());
        }
        if self.calibration.is_some() {
            return Err(
                "calibration is not consumed by the SAAQ runner; omit it or use a pack recipe"
                    .to_string(),
            );
        }
        if let Some(outputs) = &self.outputs {
            if outputs.register == Some(true) {
                return Err(
                    "saaq recipes do not register artifacts; omit outputs.register or use type 'register'"
                        .to_string(),
                );
            }
            if outputs.registry_path.is_some() {
                return Err(
                    "saaq recipes do not write a registry; omit outputs.registry_path".to_string(),
                );
            }
        }
        Ok(())
    }

    fn validate_saaq_runner_agreement(&self) -> Result<(), String> {
        if self.recipe_type != RecipeType::Saaq {
            return Ok(());
        }
        let (Some(path), Some(json)) = (&self.source_path, &self.source_json) else {
            return Ok(());
        };
        let contents = serde_json::to_string(json)
            .map_err(|e| format!("failed to serialize recipe for SAAQ validation: {e}"))?;
        let config = crate::saaq::SaaqRunConfig::from_json(&contents, path, None)?;
        crate::saaq::validate_saaq_recipe_invariants(&contents, &config)
    }

    fn validate_declared_source_format(&self) -> Result<(), String> {
        let Some(source_format) = self
            .inputs
            .as_ref()
            .and_then(|inputs| inputs.source_format.as_deref())
        else {
            return Ok(());
        };
        if !VALID_SOURCE_FORMATS.contains(&source_format) {
            return Err(format!(
                "inputs.source_format '{source_format}' must be one of: {}",
                VALID_SOURCE_FORMATS.join(", ")
            ));
        }
        self.reject_unpackable_source_format(source_format)?;
        self.reject_unexportable_source_format(source_format)
    }

    fn reject_unpackable_source_format(&self, source_format: &str) -> Result<(), String> {
        if self.recipe_type.is_pack() && !PACKABLE_SOURCE_FORMATS.contains(&source_format) {
            return Err(format!(
                "recipe type '{}' cannot pack source format '{source_format}'; \
                 the packer accepts: {}",
                self.recipe_type,
                PACKABLE_SOURCE_FORMATS.join(", ")
            ));
        }
        Ok(())
    }

    fn reject_unexportable_source_format(&self, source_format: &str) -> Result<(), String> {
        if self.recipe_type == RecipeType::GgufExport
            && !EXPORTABLE_SOURCE_FORMATS.contains(&source_format)
        {
            return Err(format!(
                "recipe type 'gguf_export' cannot convert source format '{source_format}'; \
                 existing GGUF artifacts should use type 'register'. Conversion accepts: {}",
                EXPORTABLE_SOURCE_FORMATS.join(", ")
            ));
        }
        Ok(())
    }

    fn validate_pack_lineage(&self, manifest: &Manifest) -> Result<(), String> {
        let Some(lineage) = self
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.lineage.as_ref())
        else {
            return Ok(());
        };
        if let Some(parent_manifest_id) = lineage.parent_manifest_id.as_deref()
            && parent_manifest_id != manifest.metadata.manifest_id
        {
            return Err(format!(
                "outputs.lineage.parent_manifest_id '{parent_manifest_id}' does not match the \
                 referenced manifest's metadata.manifest_id '{}'",
                manifest.metadata.manifest_id
            ));
        }
        if let Some(parent_path) = lineage.parent_path.as_deref()
            && parent_path != manifest.source_artifact.path
        {
            return Err(format!(
                "outputs.lineage.parent_path '{parent_path}' does not match the referenced \
                 manifest's source_artifact.path '{}'",
                manifest.source_artifact.path
            ));
        }
        Ok(())
    }

    fn validate_outputs(&self) -> Result<(), String> {
        let Some(outputs) = &self.outputs else {
            return Ok(());
        };
        self.validate_generated_format(outputs)?;
        self.validate_goz1_version(outputs)?;
        self.validate_checksum_algorithm(outputs)?;
        self.validate_lineage_recipe_id(outputs)?;
        self.validate_register_flags(outputs)
    }

    fn validate_generated_format(&self, outputs: &RecipeOutputs) -> Result<(), String> {
        let Some(generated_format) = outputs.generated_format.as_deref() else {
            return self.require_generated_format_when_needed();
        };
        if !VALID_GENERATED_FORMATS.contains(&generated_format) {
            return Err(format!(
                "outputs.generated_format '{generated_format}' must be one of: {}",
                VALID_GENERATED_FORMATS.join(", ")
            ));
        }
        self.generated_format_matches_type(generated_format)
    }

    fn require_generated_format_when_needed(&self) -> Result<(), String> {
        if self.recipe_type.is_pack() || self.recipe_type == RecipeType::GgufExport {
            return Err(format!(
                "recipe type '{}' requires outputs.generated_format",
                self.recipe_type
            ));
        }
        Ok(())
    }

    fn generated_format_matches_type(&self, generated_format: &str) -> Result<(), String> {
        match self.recipe_type {
            RecipeType::Goz1Pack if generated_format != "goz1" => Err(format!(
                "a goz1_pack recipe must set outputs.generated_format to 'goz1' (got '{generated_format}')"
            )),
            RecipeType::TernaryPack if !matches!(generated_format, "goz1" | "ternary") => {
                Err(format!(
                    "a ternary_pack recipe must set outputs.generated_format to 'goz1' or 'ternary' (got '{generated_format}')"
                ))
            }
            RecipeType::GgufExport if generated_format != "gguf" => Err(format!(
                "a gguf_export recipe must set outputs.generated_format to 'gguf' (got '{generated_format}')"
            )),
            _ => Ok(()),
        }
    }

    fn validate_goz1_version(&self, outputs: &RecipeOutputs) -> Result<(), String> {
        let Some(version) = outputs.goz1_version else {
            return Ok(());
        };
        if version != SUPPORTED_GOZ1_VERSION {
            return Err(format!(
                "outputs.goz1_version must be {SUPPORTED_GOZ1_VERSION} (got {version})"
            ));
        }
        if outputs.generated_format.as_deref() != Some("goz1") {
            return Err(
                "outputs.goz1_version is only meaningful when outputs.generated_format is 'goz1'"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn validate_checksum_algorithm(&self, outputs: &RecipeOutputs) -> Result<(), String> {
        match outputs.checksum_algorithm.as_deref() {
            Some(algorithm) if algorithm != "sha256" => Err(format!(
                "outputs.checksum_algorithm '{algorithm}' is not supported; use sha256"
            )),
            _ => Ok(()),
        }
    }

    fn validate_lineage_recipe_id(&self, outputs: &RecipeOutputs) -> Result<(), String> {
        match outputs
            .lineage
            .as_ref()
            .and_then(|lineage| lineage.recipe_id.as_deref())
        {
            Some(recipe_id) if recipe_id != self.recipe_id => Err(format!(
                "outputs.lineage.recipe_id '{recipe_id}' must match recipe_id '{}'",
                self.recipe_id
            )),
            _ => Ok(()),
        }
    }

    fn validate_register_flags(&self, outputs: &RecipeOutputs) -> Result<(), String> {
        if outputs.register == Some(false) && self.recipe_type == RecipeType::Register {
            return Err(
                "a register recipe exists to add its source manifest to the artifact registry, \
                 so outputs.register must not be false"
                    .to_string(),
            );
        }
        if outputs.register == Some(true) && self.recipe_type.is_pack() {
            if outputs.manifest_id.is_none() {
                return Err(
                    "outputs.register is true, so outputs.manifest_id is required to emit an artifact manifest"
                        .to_string(),
                );
            }
            if outputs.artifact_path.is_none() {
                return Err(
                    "outputs.register is true, so outputs.artifact_path is required to record the emitted artifact"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    fn validate_references(&self) -> Result<(), String> {
        self.validate_source_manifest_reference()?;
        self.validate_goz1_reference()
    }

    fn validate_source_manifest_reference(&self) -> Result<(), String> {
        let Some(reference) = self.source_manifest_ref() else {
            return Ok(());
        };
        let Some(manifest) = self.load_referenced_manifest(reference, "inputs.source_manifest")?
        else {
            return Ok(());
        };
        self.assert_declared_source_format(&manifest)?;
        self.validate_typed_source_constraints(&manifest)
    }

    fn assert_declared_source_format(&self, manifest: &Manifest) -> Result<(), String> {
        let Some(expected) = self
            .inputs
            .as_ref()
            .and_then(|inputs| inputs.source_format.as_deref())
        else {
            return Ok(());
        };
        if manifest.source_artifact.format == expected {
            Ok(())
        } else {
            Err(format!(
                "inputs.source_format '{expected}' does not match the referenced manifest's \
                 source_artifact.format '{}'",
                manifest.source_artifact.format
            ))
        }
    }

    fn validate_typed_source_constraints(&self, manifest: &Manifest) -> Result<(), String> {
        if self.recipe_type.is_pack() {
            self.reject_unpackable_source_format(&manifest.source_artifact.format)?;
            self.validate_pack_lineage(manifest)?;
        }
        if self.recipe_type == RecipeType::GgufExport {
            self.reject_unexportable_source_format(&manifest.source_artifact.format)?;
        }
        if self.recipe_type == RecipeType::Register {
            self.validate_register_manifest_identity(manifest)?;
        }
        Ok(())
    }

    fn validate_register_manifest_identity(&self, manifest: &Manifest) -> Result<(), String> {
        if let Some(manifest_id) = self
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.manifest_id.as_deref())
            && manifest_id != manifest.metadata.manifest_id
        {
            return Err(format!(
                "outputs.manifest_id '{manifest_id}' does not match the referenced \
                 manifest's metadata.manifest_id '{}'; a register recipe records an \
                 existing manifest, it does not rename one",
                manifest.metadata.manifest_id
            ));
        }
        let Some(generated_format) = self
            .outputs
            .as_ref()
            .and_then(|outputs| outputs.generated_format.as_deref())
        else {
            return Ok(());
        };
        match manifest.generated_artifact.as_ref() {
            Some(generated) if generated.format == generated_format => Ok(()),
            Some(generated) => Err(format!(
                "outputs.generated_format '{generated_format}' does not match the \
                 referenced manifest's generated_artifact.format '{}'",
                generated.format
            )),
            None => Err(format!(
                "outputs.generated_format '{generated_format}' is set but the \
                 referenced manifest has no generated_artifact to register"
            )),
        }
    }

    fn validate_goz1_reference(&self) -> Result<(), String> {
        let Some(reference) = self.goz1_ref() else {
            return Ok(());
        };
        let Some(manifest) = self.load_referenced_manifest(reference, "inputs.goz1_ref")? else {
            return Ok(());
        };
        match manifest.generated_artifact.as_ref() {
            Some(generated) if generated.format == "goz1" => Ok(()),
            Some(generated) => Err(format!(
                "inputs.goz1_ref must point at a manifest whose generated_artifact.format \
                 is 'goz1' (got '{}')",
                generated.format
            )),
            None => Err(
                "inputs.goz1_ref must point at a manifest that carries a generated_artifact"
                    .to_string(),
            ),
        }
    }
}
