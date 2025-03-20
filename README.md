<h1 align="center">Predict French TV Audience</h1>

<p align="center">
  <img width="80%" src="https://github.com/4strium/Predict-French-TV-Audience/blob/main/minia.png?raw=true" alt="Miniature project">
</p>

## Predicting TV Viewership 

This project utilizes various regressive algorithms to predict film viewership based on various features such as channel, genre, nationality, duration, IMDb ratings, airing date, and other relevant factors.

### 1. Requirements

Before running the script, ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### 2. Dataset

The model requires a dataset (`database_final.csv`) containing film details and audience numbers. 

The dataset should include columns such as:
- `Chaîne`: The channel broadcasting the film.
- `Genres`: The genres of the film.
- `Nationalité`: The nationality of the film.
- `Durée (en min.)`: The duration of the film in minutes.
- `IMDB - Note moyenne`: IMDb average rating.
- `IMDB - Nombre de votes`: Number of IMDb votes.
- `Année de sortie`: Year of release.
- `Jour`: Day of airing.
- `Mois`: Month of airing.
- `Année de diffusion`: Year of airing.
- `Vacances scolaires`: Whether it aired during school holidays (`oui`/`non`).
- `Week-end`: Whether it aired on a weekend (`0` or `1`).
- `Saison`: Season based on the month.
- `Téléspectateurs (en millions)`: Target variable - number of viewers in millions.

### 3. Steps in the Code

1. **Load the Dataset**
   - The script reads the CSV file and processes the date column.
   - It extracts relevant features like month, weekend indicator, and season.
   
2. **Data Preprocessing**
   - Genres and nationalities are split into multiple categories and one-hot encoded.
   - Categorical variables (`Chaîne`, `Jour`) are transformed using `OneHotEncoder`.
   - Numerical variables are standardized using `StandardScaler`.
   - School holidays (`Vacances scolaires`) are label-encoded.
   
3. **Model Training**
   - The dataset is split into training (80%) and testing (20%) sets.
   - An XGBoost Regressor model is trained with specific hyperparameters:
     ```python
     model = xgboost.XGBRegressor(max_depth=3, subsample=0.91, tree_method='hist', device='cuda', seed=42, n_estimators=30, learning_rate=0.32)
     ```

4. **Model Evaluation**
   - The model predicts audience numbers on the test set.
   - Performance is evaluated using RMSE (Root Mean Squared Error) and R² Score.
   - A correlation heatmap is generated to visualize feature relationships.
   - Feature importance is displayed in a bar chart.

|Name|Link to the code| RMSE (millions) | R² Score |
|:---:|:---:| :---: | :---: |
|XGBoost|[Here](https://github.com/4strium/Predict-French-TV-Audience/blob/main/XGBoost.py)|0.516|0.96|
|Random Forest|[Here](https://github.com/4strium/Predict-French-TV-Audience/blob/main/random_forest_regressor.py)|0.537|0.95|
|K-Nearest Neighbors|[Here](https://github.com/4strium/Predict-French-TV-Audience/blob/main/KNN.py)|1.506|0.64|
   
5. **Making Predictions**
   - The script includes sample data for four movies airing in 2025.
   - The model predicts their expected viewership based on their attributes.

### 4. Consideration of the importance of the different parameters

The different parameters have varying levels of importance in the final prediction. You should consider that, depending on the channel you are targeting, the result may deviate more or less from the actual measurements. A channel with a limited selection of genres can be very easy to predict, whereas, on the other hand, a channel with a lot of diversity can be much more difficult to anticipate.

That's why, if you conduct your own tests and notice very little difference between the same film being broadcast in June or December, it's completely fine—this simply means that the importance of the airing month is close to zero, for example.

Here are the top three parameters that are highly correlated with the target variable:

1) Channel
2) Number of IMDb votes
3) Genre

### 5. Running Your Own Tests

To test with your own data:

1. Modify the `database_final.csv` file with your own dataset.
2. If you want to predict viewership for new shows:
   - Define a `new_data` DataFrame in the same format as the provided examples.
   - Ensure proper encoding (genres, nationalities, categorical features).
   - Use the trained model to predict audience numbers:
     ```python
     new_predictions = model.predict(new_data_final)
     ```
   - Print or visualize the results as needed.

### 6. Output

The script prints predictions in the format:
```
Prédiction pour [Film] diffusé sur [Chaîne] un [Jour] soir de [Mois] [Année] pendant les vacances : X millions de téléspectateurs.
```

### 7. Notes
- Ensure the dataset is formatted correctly before running the script.
- Adjust hyperparameters if necessary to improve prediction accuracy.
- Experiment with additional features to enhance the model's performance.

### 8. Video
A explanatory video in french is available, just click on the picture below.


<p align="center" href="test.fr">
  <img width="80%" src="https://github.com/4strium/Predict-French-TV-Audience/blob/main/minia-youtube.png?raw=true" alt="Miniature youtube">
</p>

### 9. Sources
- CNC/Médiamétrie : Previous known viewership
- IMDb : Data on films