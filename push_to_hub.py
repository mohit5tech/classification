import argparse
from huggingface_hub import HfApi, HfFolder

class HuggingFaceUploader:
    def __init__(self, hf_token, repo_name, local_folder_path):
        self.hf_token = hf_token
        self.repo_name = repo_name
        self.local_folder_path = local_folder_path
        self.hf_folder = HfFolder()
        self.hf_api = HfApi(token=self.hf_token)
        
        self.hf_folder.save_token(self.hf_token)
        self.available_gpus = torch.cuda.device_count()
        
    def upload_folder(self, repo_type='model'):
        full_repo_name = f"mohit9999/{self.repo_name}"
        repo_url = self.hf_api.create_repo(
            repo_id=self.repo_name,
            token=self.hf_token,
            private=True, 
            repo_type=repo_type
        )
        self.hf_api.upload_folder(
            folder_path=self.local_folder_path,
            repo_id=full_repo_name,
            repo_type=repo_type,
            ignore_patterns=["*.pyc", "__pycache__"],
        )
        print(f"Files successfully uploaded to {full_repo_name}")
        
    # def print_gpu_info(self):
    #     print(f"Available GPUs: {self.available_gpus}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
    parser.add_argument("--token", required=False, help="Hugging Face API token",default="")
    parser.add_argument("--repo", required=False, help="Repository name on Hugging Face",default="xray_model")
    parser.add_argument("--path", required=True, help="Local folder path to upload")
    parser.add_argument("--type", default="model", choices=["model", "dataset", "space"], help="Repository type")
    args = parser.parse_args()
    uploader = HuggingFaceUploader(args.token, args.repo, args.path)
    # uploader.print_gpu_info()
    uploader.upload_folder(repo_type=args.type)
