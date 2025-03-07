import os
import sys
from typing import Text
import subprocess
 
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
 
PIPELINE_NAME = "faishal_ali-pipeline"
 
# pipeline inputs
DATA_ROOT = "data/youtube-comments"
DATA_PROCESSED_PATH = "data/youtube-comments/processed_comments.csv"
PREPROCESS_MODULE_FILE = "modules/sentiment_preprocess.py"
TRANSFORM_MODULE_FILE = "modules/sentiment_transform.py"
TRAINER_MODULE_FILE = "modules/sentiment_trainer.py"
TUNER_MODULE_FILE = "modules/sentiment_tuner.py"
# requirement_file = os.path.join(root, "requirements.txt")
 
# pipeline outputs
OUTPUT_BASE = "outputs"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")

def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline: # type: ignore
    
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available 
        # during execution time.
        "----direct_num_workers=1" 
    ]
    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )
    
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    # Jalankan Preprocessing sebelum pipeline
    print("🔄 Menjalankan preprocessing data...")
    try:
        result = subprocess.run(
            ["python", f"{PREPROCESS_MODULE_FILE}"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("✅ Preprocessing berhasil dijalankan!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error saat menjalankan preprocessing: {e}")
        print("📜 Log Error:")
        print(e.stderr)  
        raise
    
    # Pastikan file hasil preprocessing tersedia sebelum melanjutkan pipeline
    if not os.path.exists(DATA_PROCESSED_PATH):
        raise FileNotFoundError(f"❌ File hasil preprocessing tidak ditemukan: {DATA_PROCESSED_PATH}")

    print(f"✅ File hasil preprocessing tersedia: {DATA_PROCESSED_PATH}")
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)