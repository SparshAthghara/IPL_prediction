import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

print("Step 1: Load the dataset...")
df = pd.read_csv("IPL.csv")
df.dropna(inplace=True)

print("Step 2: Aggregate to innings-level score per team...")
match_df = df.groupby(['ID', 'innings', 'BattingTeam'])['total_run'].sum().reset_index()

# Optional: Print innings distribution for debugging
print("\nInnings Distribution:")
print(match_df['innings'].value_counts())

print("Step 3: Encode categorical team names...")
le = LabelEncoder()
# Fit and transform on the aggregated data
match_df['BattingTeam'] = le.fit_transform(match_df['BattingTeam'])

# Features and target for modeling
X = match_df[['BattingTeam', 'innings']]
y = match_df['total_run']

print("Step 4: Split into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 5: Train the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Step 6: Make predictions on the test set...")
y_pred = model.predict(X_test)

print("Step 7: Evaluate the model...")
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Evaluate accuracy within a set of margin error values
margins = [5, 10, 15, 20, 50]
print("\nAccuracy within margins of error:")
for m in margins:
    within_margin = np.abs(y_pred - y_test) <= m
    accuracy = np.mean(within_margin)
    print(f"±{m} runs: {accuracy * 100:.2f}%")

print("\nSample Predictions:")
comparison = pd.DataFrame({
    'Actual Score': y_test[:10].values,
    'Predicted Score': np.round(y_pred[:10], 2),
    'Error': np.round(y_pred[:10] - y_test[:10], 2)
})
print(comparison)

# Create a consistent team mapping based on the training data
# Invert the transformation to obtain original team names
encoded_teams = match_df['BattingTeam'].unique()
original_teams = le.inverse_transform(encoded_teams)
team_mapping = dict(zip(encoded_teams, original_teams))

print("\nEstimated Team Scores with Margin of Error and Confidence:")
print(f"{'Team':35} {'Innings':>7} {'Predicted Score':>17} {'±MAE':>10} {'Confidence':>12}")
# Set a custom margin of error to be used in predictions (can be adjusted)
custom_mae = 40
confidence_mae = np.mean(np.abs(y_pred - y_test) <= custom_mae)

for team_id, team_name in team_mapping.items():
    for inning in [1, 2]:
        sample = pd.DataFrame([[team_id, inning]], columns=['BattingTeam', 'innings'])
        predicted_score = int(model.predict(sample)[0])
        print(f"{team_name:35} {inning:7} {predicted_score:17} {custom_mae:10.2f} {confidence_mae*100:11.2f}%")

def predict_team_score(team_name, inning, mae_value):
    """
    Predicts team score and calculates confidence for a given team and inning.
    Parameters:
        team_name (str): Name of the team.
        inning (int): Inning number (1 or 2).
        mae_value (float): Margin of error for confidence calculation.
    Returns:
        predicted_score (int): The predicted score.
        confidence (float): Confidence level based on how often predictions fall within mae_value.
    """
    if team_name not in team_mapping.values():
        raise ValueError(f"Team '{team_name}' not found.")
    
    # Get the encoded team id using the reverse mapping
    team_id = list(team_mapping.keys())[list(team_mapping.values()).index(team_name)]
    sample = pd.DataFrame([[team_id, inning]], columns=['BattingTeam', 'innings'])
    predicted_score = int(model.predict(sample)[0])
    confidence = np.mean(np.abs(y_pred - y_test) <= mae_value) * 100
    return predicted_score, confidence

# Example usage:
team_name = "Sunrisers Hyderabad"
inning = 2
custom_mae = 40  # Set the margin of error value as required
predicted_score, confidence = predict_team_score(team_name, inning, custom_mae)
print(f"\nPredicted Score for {team_name} in Inning {inning}: {predicted_score}")
print(f"Confidence Level: {confidence:.2f}%")

# Save the model and necessary artifacts for future inference
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(y_pred, "y_pred.pkl")
joblib.dump(y_test, "y_test.pkl")
