from utils.fetch_data import DataFetcher
from utils.data_preprocessing import DataPreprocessor
from utils.model_finetuning import ModelFinetuner

if __name__ == "__main__":
    # Instantiate each class
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    finetuner = ModelFinetuner()

    # Run fetch step
    fetcher.run()
    # Run preprocessing step
    preprocessor.run()
    # Run finetuning step
    finetuner.run()

    print("All steps (fetch, preprocess, finetune) completed.")