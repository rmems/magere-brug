mod manifest;
mod checksum;
mod registry;

use clap::{Parser, Subcommand};
use manifest::Manifest;
use registry::ArtifactRegistry;
use std::path::Path;

#[derive(Parser)]
#[command(name = "magere")]
#[command(about = "magere-brug model quantization manifest CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a manifest against the JSON schema
    Validate {
        /// Path to manifest JSON file
        #[arg(value_name = "FILE")]
        path: std::path::PathBuf,
    },
    /// Register a model from its manifest
    Register {
        /// Path to manifest JSON file
        #[arg(value_name = "FILE")]
        manifest: std::path::PathBuf,

        /// Path to save registry (optional)
        #[arg(short, long)]
        registry: Option<std::path::PathBuf>,
    },
    /// Inspect a manifest
    Inspect {
        /// Path to manifest JSON file
        #[arg(value_name = "FILE")]
        path: std::path::PathBuf,
    },
    /// Verify artifact checksum
    Verify {
        /// Path to artifact file
        #[arg(value_name = "FILE")]
        artifact: std::path::PathBuf,

        /// Expected SHA256 checksum
        #[arg(value_name = "SHA256")]
        checksum: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { path } => {
            match validate_command(&path) {
                Ok(msg) => println!("{}", msg),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Register { manifest, registry } => {
            match register_command(&manifest, registry.as_deref()) {
                Ok(msg) => println!("{}", msg),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Inspect { path } => {
            match inspect_command(&path) {
                Ok(msg) => println!("{}", msg),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Verify { artifact, checksum } => {
            match verify_command(&artifact, &checksum) {
                Ok(msg) => println!("{}", msg),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

fn validate_command(path: &Path) -> Result<String, String> {
    let manifest = Manifest::from_file(path)
        .map_err(|e| format!("Failed to load manifest: {}", e))?;

    manifest.validate()?;

    Ok(format!(
        "✓ Manifest '{}' is valid (model: {}, family: {})",
        manifest.metadata.manifest_id, manifest.model.name, manifest.model.family
    ))
}

fn register_command(
    manifest_path: &Path,
    registry_path: Option<&Path>,
) -> Result<String, String> {
    let manifest = Manifest::from_file(manifest_path)
        .map_err(|e| format!("Failed to load manifest: {}", e))?;

    manifest.validate()?;

    let registry_file = registry_path.unwrap_or_else(|| Path::new("registry.json"));
    let mut registry = if registry_file.exists() {
        let content = std::fs::read_to_string(registry_file)
            .map_err(|e| format!("Failed to read registry: {}", e))?;
        ArtifactRegistry::from_json(&content)
            .map_err(|e| format!("Failed to parse registry: {}", e))?
    } else {
        ArtifactRegistry::new()
    };

    registry.register(&manifest)?;

    let serialized = registry
        .to_json_pretty()
        .map_err(|e| format!("Failed to serialize registry: {}", e))?;
    std::fs::write(registry_file, serialized)
        .map_err(|e| format!("Failed to write registry: {}", e))?;

    Ok(format!(
        "✓ Registered model '{}' (slug: {}) to {}",
        manifest.model.name,
        manifest.model.slug,
        registry_file.display()
    ))
}

fn inspect_command(path: &Path) -> Result<String, String> {
    let manifest = Manifest::from_file(path)
        .map_err(|e| format!("Failed to load manifest: {}", e))?;

    let mut output = String::new();
    output.push_str(&format!("Manifest ID: {}\n", manifest.metadata.manifest_id));
    output.push_str(&format!("Model: {} ({})\n", manifest.model.name, manifest.model.family));
    output.push_str(&format!("Slug: {}\n", manifest.model.slug));
    output.push_str(&format!("Architecture: {}\n", manifest.model.architecture));
    output.push_str(&format!(
        "Parameters: {} active",
        manifest.model.parameter_count.active
    ));

    if let Some(total) = manifest.model.parameter_count.total {
        output.push_str(&format!(", {} total", total));
    }
    output.push('\n');

    output.push_str(&format!("Source Format: {}\n", manifest.source_artifact.format));
    output.push_str(&format!("Source Path: {}\n", manifest.source_artifact.path));

    if let Some(url) = &manifest.source_artifact.source_url {
        output.push_str(&format!("Source URL: {}\n", url));
    }

    if let Some(checksum) = &manifest.source_artifact.checksum {
        if let Some(sha256) = &checksum.sha256 {
            output.push_str(&format!("SHA256: {}\n", sha256));
        }
    }

    if let Some(generated) = &manifest.generated_artifact {
        output.push_str(&format!("Generated Format: {}\n", generated.format));
        if let Some(path) = &generated.path {
            output.push_str(&format!("Generated Path: {}\n", path));
        }
        if let Some(status) = &generated.status {
            output.push_str(&format!("Generated Status: {}\n", status));
        }
    }

    if let Some(quant) = &manifest.quantization {
        if let Some(method) = &quant.method {
            output.push_str(&format!("Quantization Method: {}\n", method));
        }
        if let Some(bits) = quant.bits {
            output.push_str(&format!("Quantization Bits: {}\n", bits));
        }
    }

    Ok(output)
}

fn verify_command(artifact: &Path, expected_checksum: &str) -> Result<String, String> {
    let valid = checksum::verify_checksum(artifact, expected_checksum)
        .map_err(|e| format!("Failed to verify checksum: {}", e))?;

    if valid {
        Ok(format!("✓ Checksum verified for {}", artifact.display()))
    } else {
        Err(format!(
            "✗ Checksum mismatch for {}",
            artifact.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parser_validate() {
        let args = vec!["magere", "validate", "/path/to/manifest.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parser_inspect() {
        let args = vec!["magere", "inspect", "/path/to/manifest.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parser_verify() {
        let args = vec!["magere", "verify", "/path/to/artifact", "abc123"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parser_register() {
        let args = vec!["magere", "register", "/path/to/manifest.json", "--registry", "/path/to/registry.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }
}
