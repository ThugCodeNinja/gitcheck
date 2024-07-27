from google.cloud import aiplatform

project = 'lloyds-hack-grp-43'
location = 'us-east1'  # e.g., 'us-central1'
model_display_name = 'xgboost_model'
gcs_source_uri = 'gs://lloyds-hyderabad-hackathon/lloyds-hack-grp-43/model.pkl'
serving_container_image_uri = 'us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-3:latest'

aiplatform.init(project=project, location=location)

model = aiplatform.Model.upload(
    display_name=model_display_name,
    artifact_uri=gcs_source_uri,
    serving_container_image_uri=serving_container_image_uri,
    serving_container_predict_route='/predict',
    serving_container_health_route='/health'
)

model.wait()
