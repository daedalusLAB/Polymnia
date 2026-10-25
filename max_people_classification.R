#' max_people_classification.R
#'
#' This script processes OpenPose JSON keypoints using the `multimolang::dfMaker()` function.
#' It standardizes the coordinate system, detects the number of visible persons in the clip,
#' and exports the result as a `.parquet` file. It also logs diagnostic information and 
#' installs any missing R packages locally inside `create_datasets/R_libs`.
#'
#' Inputs:
#'   - `input_folder`: Path to the folder containing `JSON_FILES` from OpenPose.
#'   - `expression_year_dir`: Path to the output `dataset/` directory.
#'
#' Outputs:
#'   - A `.parquet` file stored in `dataset/{n_persons}_persons/parquet_files/`.
#'   - A `num_persons.txt` file stored in the input folder for downstream use.
#'
#' Dependencies:
#'   - R packages: `multimolang`, `arrow`, `logger`
#'   - Configuration file: `config_dfMaker.json` must be present in the working directory.
#'
#' Usage (via argos.py):
#'   Rscript max_people_classification.R path/to/JSON_FILES/ path/to/dataset/

# Get the directory where this Python script resides
python_script_dir <- normalizePath(Sys.getenv("PYTHON_SCRIPT_DIR"), mustWork = FALSE)

# Define a relative path for local R libraries inside create_datasets
user_lib_path <- file.path(python_script_dir, "R_libs")

# Create the directory if it doesn't exist
if (!dir.exists(user_lib_path)) {
  dir.create(user_lib_path, recursive = TRUE)
}

# Set R to use this local library path
.libPaths(user_lib_path)

# Function to check and install packages if missing
check_and_install <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, lib = user_lib_path, dependencies = TRUE)
  }
  library(package, character.only = TRUE)
}

# Load required packages
check_and_install("multimolang")
check_and_install("logger")
check_and_install("arrow") 

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_folder <- args[1]
expression_year_dir <- args[2]

log_info("Input folder: {input_folder}")
log_info("Expression year directory: {expression_year_dir}")

# dfMaker is now loaded directly from the 'multimolang' package (no .rda file required)
log_info("multimolang package loaded successfully")

# Run dfMaker to process the folder
tryCatch({
  log_info("Starting dfMaker processing")
  log_info("Contents of input folder: {paste(list.files(input_folder), collapse=', ')}")
  
  clip <- dfMaker(
    input.folder = input_folder,
    no_save = TRUE,
    config.path = "config_dfMaker.json"
  )
  
  log_info("dfMaker processing completed")
  log_info("Structure of clip: {capture.output(str(clip))}")
  
}, error = function(e) {
  log_error("Error in dfMaker: {conditionMessage(e)}")
  log_error("Traceback: {paste(capture.output(traceback()), collapse='\n')}")
  stop("Execution interrupted due to an error in dfMaker")
})

# Extract clip ID
clip$id
# Un clip sin una sola deteccion (OpenPose devolvio {"people": []} en todos los frames)
# hacia que dfMaker devolviera NULL y `write_parquet(NULL)` reventara, tumbando el lote entero
# en la etapa de limpieza. Se descarta aqui: num_persons.txt = 0, sin parquet y exit 0. La
# etapa 6 cuenta esos clips (tools/descartes_sin_pose.py) y aplica su umbral.
sin_pose <- is.null(clip) || nrow(clip) == 0
if (!sin_pose) {
  num_people <- suppressWarnings(max(clip$people_id))
  sin_pose <- !is.finite(num_people)
} else {
  num_people <- 0
}
if (sin_pose) {
  log_info("sin pose: el clip no tiene ninguna deteccion; se descarta sin escribir parquet")
  write(0, file = file.path(input_folder, "num_persons.txt"))
  cat("SIN_POSE", basename(dirname(input_folder)), "\n")
  quit(save = "no", status = 0)
}

# Define output path based on expression_year_dir
output_path <- file.path(
  expression_year_dir,
  paste0(num_people, "_persons"),
  "parquet_files",
  paste0(unique(clip$id), ".parquet")
)

# Create the output directory if it doesn't exist
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write the processed clip to a Parquet file
write_parquet(x = clip, sink = output_path)

# Save the number of people to a file for future use
num_people_file <- file.path(input_folder, "num_persons.txt")
write(num_people, file = num_people_file)

cat("Processed file saved at:", output_path, "\n")
