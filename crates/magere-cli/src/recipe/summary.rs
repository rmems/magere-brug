use super::{Handoff, Recipe, RecipeOutputs};

impl Recipe {
    /// Human-readable rendering used by `magere recipe inspect`.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        append_header(self, &mut out);
        append_inputs(self, &mut out);
        if let Some(outputs) = &self.outputs {
            append_outputs(self, outputs, &mut out);
        }
        append_calibration(self, &mut out);
        if let Some(handoff) = &self.handoff {
            append_handoff(handoff, &mut out);
        }
        out
    }
}

fn append_header(recipe: &Recipe, out: &mut String) {
    out.push_str(&format!("Recipe ID: {}\n", recipe.recipe_id));
    out.push_str(&format!("Type: {}\n", recipe.recipe_type));
    if let Some(description) = &recipe.description {
        out.push_str(&format!("Description: {description}\n"));
    }
    out.push_str(&format!("Runner: {}\n", recipe.runner_owner()));
}

fn append_inputs(recipe: &Recipe, out: &mut String) {
    let Some(inputs) = &recipe.inputs else {
        return;
    };
    if let Some(source_manifest) = &inputs.source_manifest {
        out.push_str(&format!("Source Manifest: {source_manifest}\n"));
    }
    if let Some(source_format) = &inputs.source_format {
        out.push_str(&format!("Source Format: {source_format}\n"));
    }
    if let Some(goz1_ref) = &inputs.goz1_ref {
        out.push_str(&format!("GOZ1 Ref: {goz1_ref}\n"));
    }
}

fn append_outputs(recipe: &Recipe, outputs: &RecipeOutputs, out: &mut String) {
    if let Some(generated_format) = &outputs.generated_format {
        out.push_str(&format!("Generated Format: {generated_format}\n"));
    }
    if let Some(manifest_id) = &outputs.manifest_id {
        out.push_str(&format!("Output Manifest ID: {manifest_id}\n"));
    }
    if let Some(artifact_path) = &outputs.artifact_path {
        out.push_str(&format!("Output Artifact Path: {artifact_path}\n"));
    }
    if let Some(output_dir) = &outputs.output_dir {
        out.push_str(&format!("Output Dir: {output_dir}\n"));
    }
    if let Some(version) = outputs.goz1_version {
        out.push_str(&format!("GOZ1 Version: {version}\n"));
    }
    if let Some(algorithm) = &outputs.checksum_algorithm {
        out.push_str(&format!("Checksum Algorithm: {algorithm}\n"));
    }
    out.push_str(&format!(
        "Register Output: {}\n",
        recipe.registers_output(outputs)
    ));
    if let Some(registry_path) = &outputs.registry_path {
        out.push_str(&format!("Registry Path: {registry_path}\n"));
    }
    append_lineage(outputs, out);
}

fn append_lineage(outputs: &RecipeOutputs, out: &mut String) {
    let Some(lineage) = &outputs.lineage else {
        return;
    };
    if let Some(parent_manifest_id) = &lineage.parent_manifest_id {
        out.push_str(&format!("Lineage Parent Manifest: {parent_manifest_id}\n"));
    }
    if let Some(parent_path) = &lineage.parent_path {
        out.push_str(&format!("Lineage Parent Path: {parent_path}\n"));
    }
    if let Some(recipe_id) = &lineage.recipe_id {
        out.push_str(&format!("Lineage Recipe: {recipe_id}\n"));
    }
}

fn append_calibration(recipe: &Recipe, out: &mut String) {
    let Some(calibration) = &recipe.calibration else {
        return;
    };
    out.push_str(&format!("Calibration Dataset: {}\n", calibration.dataset));
    if let Some(dataset_path) = &calibration.dataset_path {
        out.push_str(&format!("Calibration Dataset Path: {dataset_path}\n"));
    }
    if let Some(config_path) = &calibration.config_path {
        out.push_str(&format!("Calibration Config Path: {config_path}\n"));
    }
    if let Some(sample_count) = calibration.sample_count {
        out.push_str(&format!("Calibration Samples: {sample_count}\n"));
    }
    if let Some(seed) = calibration.seed {
        out.push_str(&format!("Calibration Seed: {seed}\n"));
    }
}

fn append_handoff(handoff: &Handoff, out: &mut String) {
    for (name, target) in handoff.targets() {
        let status = target.status.as_deref().unwrap_or("placeholder");
        let enabled = target.enabled.unwrap_or(false);
        out.push_str(&format!(
            "Handoff [{name}]: status={status}, enabled={enabled} (forward-declared; executed downstream)\n"
        ));
        if let Some(kernel_types) = &target.kernel_types {
            out.push_str(&format!(
                "Handoff [{name}] kernels: {}\n",
                kernel_types.join(", ")
            ));
        }
        if let Some(pipeline_id) = &target.pipeline_id {
            out.push_str(&format!("Handoff [{name}] pipeline: {pipeline_id}\n"));
        }
        if let Some(notes) = &target.notes {
            out.push_str(&format!("Handoff [{name}] notes: {notes}\n"));
        }
    }
}
