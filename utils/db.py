import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from datetime import datetime
import json
import streamlit as st
from decimal import Decimal
from utils.error_handling import handle_db_errors
import ntplib
import time
import logging

logger = logging.getLogger(__name__)

def check_time_sync():
    """Check if system time is synchronized with NTP server"""
    try:
        ntp_client = ntplib.NTPClient()
        response = ntp_client.request('pool.ntp.org')
        system_time = time.time()
        ntp_time = response.tx_time
        offset = abs(system_time - ntp_time)
        
        if offset > 30:  # More than 30 seconds difference
            logger.error(f"System time is out of sync. Offset: {offset} seconds")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check time synchronization: {str(e)}")
        return False

def convert_floats_to_decimal(obj):
    """Recursively convert float values to Decimal for DynamoDB compatibility"""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(v) for v in obj]
    return obj

class DynamoDBManager:
    def __init__(self, table_name='flux_models'):
        try:
            # Check time synchronization
            if not check_time_sync():
                st.error("""
                System time is not synchronized with AWS servers. 
                Please synchronize your system time or use an NTP service.
                
                On Linux: sudo ntpdate pool.ntp.org
                On Windows: Open Date & Time settings and click "Sync now"
                """)
            
            # Get AWS credentials from Streamlit secrets
            if 'aws' not in st.secrets:
                raise ValueError("AWS credentials not found in secrets")
            
            aws_config = st.secrets["aws"]
            required_keys = ['aws_access_key_id', 'aws_secret_access_key', 'region']
            
            # Validate all required AWS keys are present
            missing_keys = [key for key in required_keys if key not in aws_config]
            if missing_keys:
                raise ValueError(f"Missing required AWS credentials: {', '.join(missing_keys)}")
            
            # Initialize DynamoDB client with credentials and additional config
            config = Config(
                connect_timeout=5,
                read_timeout=5,
                retries={'max_attempts': 3}
            )
            
            self.dynamodb = boto3.resource(
                'dynamodb',
                aws_access_key_id=aws_config['aws_access_key_id'],
                aws_secret_access_key=aws_config['aws_secret_access_key'],
                region_name=aws_config['region'],
                config=config
            )
            self.table_name = table_name
            self.table = self.dynamodb.Table(table_name)
            
        except Exception as e:
            logger.error(f"DynamoDB initialization error: {str(e)}")
            st.error(f"""
            Error initializing DynamoDB connection. Please check:
            1. Your system time is synchronized
            2. AWS credentials are correct
            3. Network connection is stable
            
            Your .streamlit/secrets.toml file should contain:
            ```toml
            [aws]
            aws_access_key_id = "your_access_key"
            aws_secret_access_key = "your_secret_key"
            region = "your_region"
            ```
            
            Error details: {str(e)}
            """)
            raise
    
    def create_table_if_not_exists(self):
        """Create the DynamoDB table if it doesn't exist"""
        try:
            # Check if table exists
            self.dynamodb.meta.client.describe_table(TableName=self.table_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                try:
                    # Create table
                    self.table = self.dynamodb.create_table(
                        TableName=self.table_name,
                        KeySchema=[
                            {
                                'AttributeName': 'model_id',
                                'KeyType': 'HASH'  # Partition key
                            }
                        ],
                        AttributeDefinitions=[
                            {
                                'AttributeName': 'model_id',
                                'AttributeType': 'S'
                            }
                        ],
                        ProvisionedThroughput={
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    )
                    # Wait for table creation
                    self.table.meta.client.get_waiter('table_exists').wait(
                        TableName=self.table_name
                    )
                    return True
                except Exception as create_error:
                    st.error(f"Error creating DynamoDB table: {str(create_error)}")
                    return False
            else:
                st.error(f"Error checking DynamoDB table: {str(e)}")
                return False
    
    def save_model(self, model_data):
        """Save model information to DynamoDB"""
        try:
            # Validate required fields
            required_fields = ['owner', 'name', 'visibility', 'description', 'replicate_id']
            missing_fields = [field for field in required_fields if field not in model_data]
            if missing_fields:
                st.error(f"Missing required fields: {', '.join(missing_fields)}")
                return False
            
            advanced_options = model_data.get('advanced_parameters', {})
            
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return False
            
            # Convert any float values to Decimal
            advanced_options = convert_floats_to_decimal(advanced_options)
            
            item = {
                'model_id': f"{model_data['owner']}/{model_data['name']}",
                'owner': model_data['owner'],
                'name': model_data['name'],
                'visibility': model_data['visibility'],
                'description': model_data['description'],
                'replicate_id': model_data['replicate_id'],
                'version_id': model_data.get('version_id', ''),
                'advanced_parameters': advanced_options,
                'status': model_data.get('status', 'training'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'model_path': model_data.get('model_path', ''),
                'model_version': model_data.get('model_version', '')
            }
            
            # Convert any remaining float values in the entire item
            item = convert_floats_to_decimal(item)
            
            self.table.put_item(Item=item)
            return True
        except Exception as e:
            st.error(f"Error saving model to DynamoDB: {str(e)}")
            return False
    
    @handle_db_errors
    def get_all_models(self):
        """Retrieve all models from DynamoDB"""
        try:
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return []
                
            response = self.table.scan()
            return response.get('Items', [])
        except Exception as e:
            st.error(f"Error retrieving models from DynamoDB: {str(e)}")
            return []
    
    def get_model_by_id(self, model_id):
        """Retrieve a specific model by ID"""
        try:
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return None
                
            response = self.table.get_item(Key={'model_id': model_id})
            return response.get('Item')
        except Exception as e:
            st.error(f"Error retrieving model from DynamoDB: {str(e)}")
            return None
    
    def update_model_status(self, model_id, status, version_id=None):
        """Update the status of a model"""
        try:
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return False
                
            update_expr = "SET #status = :status"
            expr_names = {"#status": "status"}
            expr_values = {":status": status}
            
            if version_id:
                update_expr += ", version_id = :version_id"
                expr_values[":version_id"] = version_id
            
            self.table.update_item(
                Key={'model_id': model_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values
            )
            return True
        except Exception as e:
            st.error(f"Error updating model status in DynamoDB: {str(e)}")
            return False
    
    def get_available_models(self):
        """Get all models that are ready for use"""
        try:
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return []
                
            response = self.table.scan(
                FilterExpression='#status = :status',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':status': 'ready'}
            )
            return response.get('Items', [])
        except Exception as e:
            st.error(f"Error retrieving available models from DynamoDB: {str(e)}")
            return []
    
    @handle_db_errors
    def update_model_training_status(self, model_id, training_status, version_id, output_url):
        """Update the training status and details of a model"""
        try:
            # Create table if it doesn't exist
            if not self.create_table_if_not_exists():
                return False
            
            update_expr = "SET #status = :status, updated_at = :updated_at"
            expr_names = {"#status": "status"}
            expr_values = {
                ":status": training_status,
                ":updated_at": datetime.now().isoformat()
            }
            
            if version_id:
                update_expr += ", version_id = :version_id"
                expr_values[":version_id"] = version_id
            
            if output_url:
                update_expr += ", output_url = :output_url"
                expr_values[":output_url"] = output_url
            
            # Convert any float values to Decimal
            expr_values = convert_floats_to_decimal(expr_values)
            
            self.table.update_item(
                Key={'model_id': model_id},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values
            )
            return True
        except Exception as e:
            st.error(f"Error updating model status in DynamoDB: {str(e)}")
            return False
    
    @handle_db_errors
    def get_total_images_generated(self):
        # Implementation
        pass
    
    @handle_db_errors
    def get_available_models_count(self):
        # Implementation
        pass
    
    @handle_db_errors
    def get_active_users_count(self):
        # Implementation
        pass 