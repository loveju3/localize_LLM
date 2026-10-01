import boto3
from botocore.config import Config


class ObjectStore:
    def __init__(self, settings):
        self.settings = settings

    def client(self):
        self.settings.require("r2_endpoint_url", "r2_access_key_id", "r2_secret_access_key", "r2_bucket_name")
        return boto3.client("s3", endpoint_url=self.settings.r2_endpoint_url,
            aws_access_key_id=self.settings.r2_access_key_id.get_secret_value(),
            aws_secret_access_key=self.settings.r2_secret_access_key.get_secret_value(),
            region_name="auto", config=Config(signature_version="s3v4", connect_timeout=10,
                                             read_timeout=60, retries={"max_attempts": 2}))

    def put(self, key, data, content_type="application/octet-stream"):
        self.client().put_object(Bucket=self.settings.r2_bucket_name, Key=key,
                                 Body=data, ContentType=content_type)

    def get(self, key):
        body = self.client().get_object(Bucket=self.settings.r2_bucket_name, Key=key)["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def download_url(self, key):
        return self.client().generate_presigned_url("get_object",
            Params={"Bucket": self.settings.r2_bucket_name, "Key": key}, ExpiresIn=300)
