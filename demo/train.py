from fusion_model import FusionModel

csv_file_path = "Output/training_dateset.csv"

def main():
    fm = FusionModel(csv_file_path, hidden=64, lr=1e-3)
    fm.train(num_epochs=50, patience=5)
    fm.evaluate()


if __name__ == "__main__":
    main()

