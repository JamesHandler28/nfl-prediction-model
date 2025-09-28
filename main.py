'''=========================================================================
NFL Game Outcome Predictor

James Handler

This script trains a machine learning model (XGBoost) to predict the winner of
NFL games based on historical data. It includes feature engineering for team
strength and form, and uses a chronological split for honest evaluation.
Finally, it predicts the outcomes of manually specified upcoming games.
============================================================================'''

# --- SETUP AND DATA LOADING ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load the historical game data
dataFrame = pd.read_csv('spreadspoke_scores.csv')


# --- INITIAL DATA CLEANING AND PREPARATION ---

# Define a mapping for playoff weeks to numeric values
week_mapping = {"Wildcard": 19, "Division": 20, "Conference": 21, "Superbowl": 22}

# Select only the columns we need for the model
df_subset = dataFrame[['schedule_date', 'schedule_season', 'schedule_week',
                       'schedule_playoff', 'team_home', 'team_away',
                       'score_home', 'score_away']]

# Drop rows with missing scores (future games or errors)
df_clean = df_subset.dropna()

# Create the target variable: 1 if the home team won, 0 otherwise
df_clean['home_team_win'] = (df_clean['score_home'] > df_clean['score_away']).astype(int)

# Make sure our data is sorted chronologically
df_clean = df_clean.sort_values('schedule_date').reset_index(drop=True)


# --- FEATURE ENGINEERING ---

# Create a single list of all game appeaerances for each team for rolling stats
# Create seperate dataframes for home and away appearances
home = df_clean[['schedule_date', 'team_home', 'home_team_win']].rename(
    columns={'team_home': 'team', 'home_team_win': 'won'})
away = df_clean[['schedule_date', 'team_away', 'home_team_win']].rename(
    columns={'team_away': 'team'})
# For away teams, a home win (1) is a loss (0), so we invert the results
away['won'] = 1 - away['home_team_win']

# Combine games into a long DataFrame sorted by date
all_games = pd.concat([home, away]).sort_values('schedule_date').reset_index(drop=True)

# Calculate rolling "form" based on win percentage of the last 5 games
# .shift() helps prevent data leakage (only uses data from before current game)
all_games['rolling_form'] = all_games.groupby('team')['won'].transform(
    lambda x: x.shift().rolling(window = 5, min_periods=1).mean()
)

# Calculate rolling "strength" based on win percentage of all time
all_games['rolling_strength'] = all_games.groupby('team')['won'].transform(
    lambda x: x.shift().expanding().mean())

# Merge the new "form" feature back into the main DataFrame
df_clean = pd.merge(df_clean, all_games[['schedule_date', 'team', 'rolling_form']],
                    left_on=['schedule_date', 'team_home'], right_on=['schedule_date', 'team'],
                    how='left').rename(columns={'rolling_form': 'home_form'})
df_clean = pd.merge(df_clean, all_games[['schedule_date', 'team', 'rolling_form']],
                    left_on=['schedule_date', 'team_away'], right_on=['schedule_date', 'team'],
                    how='left').rename(columns={'rolling_form': 'away_form'})

# Clean up extra columns from the 'form' merge
df_clean = df_clean.drop(columns=['team_x', 'team_y'])
df_clean['home_form'].fillna(0.5, inplace=True)
df_clean['away_form'].fillna(0.5, inplace=True)

# Merge the new "strength" feature back into main DataFrame
df_clean = pd.merge(df_clean, all_games[['schedule_date', 'team', 'rolling_strength']],
                    left_on=['schedule_date', 'team_home'], right_on=['schedule_date', 'team'],
                    how='left').rename(columns={'rolling_strength': 'home_strength'})

df_clean = pd.merge(df_clean, all_games[['schedule_date', 'team', 'rolling_strength']],
                    left_on=['schedule_date', 'team_away'], right_on=['schedule_date', 'team'],
                    how='left').rename(columns={'rolling_strength': 'away_strength'})

# Clean up extra columns from the merges and fill initial NaN values
df_clean = df_clean.drop(columns=['team_x', 'team_y'])
df_clean.fillna(0.5, inplace=True)

# Clean the 'schedule_week' column using the mapping
df_clean['schedule_week'] = df_clean['schedule_week'].replace(week_mapping)
df_clean['schedule_week'] = pd.to_numeric(df_clean['schedule_week'])

# Convert categorical team names into a numeric format the model can use
df_encoded = pd.get_dummies(df_clean, columns=['team_home', 'team_away'])


# --- MODEL PREPARATION ---

# Define the features (X) and the target (y)
y = df_encoded['home_team_win']
X = df_encoded.drop(columns=['home_team_win', 'score_home', 'score_away'])

# Make sure data is sorted by date before splitting
X = X.sort_values('schedule_date')
y = y.loc[X.index]

# Set the cutoff point (-1 for each year before 2025)
split_point = X['schedule_season'].max() - 1

# Create the training set  (all data before the last season)
X_train = X[X['schedule_season'] < split_point]
y_train = y[X['schedule_season'] < split_point]

# Create the testing set (only the last season)
X_test = X[X['schedule_season'] >= split_point]
y_test = y[X['schedule_season'] >= split_point]

# Drop the season column after using it for the split
X_train = X_train.drop(columns=['schedule_season', 'schedule_date'])
X_test = X_test.drop(columns=['schedule_season', 'schedule_date'])


# --- MODEL TRAINING AND EVALUATION ---

# Initialize and train the XGBoost model with tuned hyperparameters
model = XGBClassifier(n_estimators = 500, learning_rate = 0.05, max_depth = 4, use_label_encoder = False, eval_metric = 'logloss')
model.fit(X_train, y_train)

# Make predictions on the test set (the last season)
predictions = model.predict(X_test)

# Calculate and print the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# --- PREDICTION ON UPCOMING GAMES ---

print("\n--- Making predictions on upcoming games ---")

# Create lookup maps from the full historical data to get the latest stats
strength_map = all_games.groupby('team')['rolling_strength'].last().to_dict()
form_map = all_games.groupby('team')['rolling_form'].last().to_dict()

# Manually define the upcoming games to predict
upcoming_games_data = {
    'team_home': [
        'Pittsburgh Steelers', 'Atlanta Falcons', 'Buffalo Bills', 'Detroit Lions',
        'Houston Texans', 'New England Patriots', 'New York Giants', 'Tampa Bay Buccaneers',
        'Los Angeles Rams', 'San Francisco 49ers', 'Kansas City Chiefs', 'Las Vegas Raiders',
        'Dallas Cowboys', 'Miami Dolphins', 'Denver Broncos'
    ],
    'team_away': [
        'Minnesota Vikings', 'Washington Commanders', 'New Orleans Saints', 'Cleveland Browns',
        'Tennessee Titans', 'Carolina Panthers', 'Los Angeles Chargers', 'Philadelphia Eagles',
        'Indianapolis Colts', 'Jacksonville Jaguars', 'Baltimore Ravens', 'Chicago Bears',
        'Green Bay Packers', 'New York Jets', 'Cincinnati Bengals'
    ]
}
predict_df = pd.DataFrame(upcoming_games_data)

# Apply the same featuer engineering pipeline to this new data
predict_df['schedule_season'] = 2025
predict_df['schedule_week'] = 4
predict_df['schedule_playoff'] = 0
predict_df['home_strength'] = predict_df['team_home'].map(strength_map).fillna(0.5)
predict_df['away_strength'] = predict_df['team_away'].map(strength_map).fillna(0.5)
predict_df['home_form'] = predict_df['team_home'].map(form_map).fillna(0.5)
predict_df['away_form'] = predict_df['team_away'].map(form_map).fillna(0.5)

# One-hot encode and align columns to match the training data exactly
predict_df_encoded = pd.get_dummies(predict_df, columns=['team_home', 'team_away'])
final_predict_df = predict_df_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make and display the final predictions
final_predictions = model.predict(final_predict_df)
final_prediction_probs = model.predict_proba(final_predict_df)

# Add results to the dataframe for a clean report
predict_df['predicted_winner'] = ['Home' if pred == 1 else 'Away' for pred in final_predictions]
predict_df['win_probability'] = [f"{prob[pred]*100:.1f}%" for pred, prob in zip(final_predictions, final_prediction_probs)]

# Determine the name of the predicted winner for the report
predict_df['predicted_winner_name'] = predict_df.apply(
    lambda row: row['team_home'] if row['predicted_winner'] == 'Home' else row['team_away'],
    axis=1
)

# Get just the last word of the winner's name
predict_df['predicted_winner_name'] = predict_df['predicted_winner_name'].apply(
    lambda full_name: full_name.split(' ')[-1]
)

# Print the final report
print("\n--- Upcoming Games Predictions ---")
print(predict_df[['team_home', 'team_away', 'predicted_winner_name', 'win_probability']])