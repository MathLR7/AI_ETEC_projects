import src


def main():
    print("Loading data...")
    dataset, cycle_of_life = src.get_data()

    print("Creating good/bad labels...")
    labeled_df = src.get_quantile(cycle_of_life)

    print("Extracting battery features...")
    features = src.extract_battery_features(dataset)

    print("Filtering features to labeled cells...")
    filtered = src.filtered_features(features, labeled_df)

    print("Creating PCA features...")
    X_train_pca, X_test_pca, y_train, y_test = src.create_pca(filtered, random_seed=42)

    print("Training and evaluating model...")
    results = src.supervised_learning(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        plot=True
    )

    knn_results = results["KNN"]
    knn_results.to_csv("knn_test_predictions.csv")

    print("Saved prediction plot to knn_predictions_pca.png")
    print("Saved test predictions to knn_test_predictions.csv")
    print(knn_results)


if __name__ == "__main__":
    main()