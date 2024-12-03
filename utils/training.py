import streamlit as st
import replicate
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def check_training_status(training_id):
    """Check the status of a training job on Replicate"""
    try:
        training = replicate.trainings.get(training_id)
        
        # Get the version ID from the training output if available
        version_id = None
        if training.status == 'succeeded' and training.version:
            # Extract only the ID part after the colon
            version_id = training.version.split(':')[-1] if ':' in training.version else training.version
            
        return {
            'status': training.status,
            'error': training.error if hasattr(training, 'error') else None,
            'version_id': version_id,
            'urls': training.urls if hasattr(training, 'urls') else None
        }
    except Exception as e:
        st.error(f"Error checking training status: {str(e)}")
        return None

def update_all_training_status(db):
    """Update the training status of all models in the database"""
    try:
        if db is None:
            st.error("Database connection is not available")
            return []
            
        models = db.get_all_models()
        status_updates = []
        
        for model in models:
            if model['status'] in ['training', 'created', 'processing']:
                training_params = model.get('training_params', {})
                training_id = training_params.get('training_id')
                
                if training_id:
                    status = check_training_status(training_id)
                    if status:
                        current_status = status['status']
                        version_id = status['version_id']
                        
                        if current_status != model['status'] or version_id:
                            try:
                                db.update_model_training_status(
                                    model_id=model['model_id'],
                                    training_status=current_status,
                                    version_id=version_id,
                                    output_url=status.get('urls', {}).get('get')
                                )
                                status_updates.append({
                                    'model': model['name'],
                                    'old_status': model['status'],
                                    'new_status': current_status,
                                    'version_id': version_id if version_id else 'N/A'
                                })
                            except ClientError as e:
                                logger.error(f"Failed to update model status: {str(e)}")
                                st.warning(f"Failed to update status for model {model['name']}")
                        
                        if current_status == 'failed' and status['error']:
                            st.error(f"Training failed for {model['name']}: {status['error']}")
        
        return status_updates
    except Exception as e:
        logger.error(f"Error in update_all_training_status: {str(e)}")
        st.error("Failed to update training status")
        return [] 