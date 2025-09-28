# NFL Game Predictor: An AI Project

This was my first big AI project where I wanted to see if I could predict the winner of an NFL game. The goal was to build a machine learning model from scratch that uses historical game data to predict if the home or away team will win.

It turned out to be a lot more than just plugging in data. I had to figure out how to clean the data, create my own features, and choose the right way to test the model to get an honest score.

---

### The Journey & Key Features

Here is a breakdown of what I built and what I learned along the way:

* **Data Cleaning:** I started with a huge CSV file of game data going all the way back to 1966. The first step was to clean it by dropping columns I didn't need and getting rid of games with missing scores.

* **Feature Engineering:** I couldn't just use the raw data, so I created my own 'clues' for the model to use.
    * **Team Strength Score:** At first, I just calculated a simple win percentage for each team. But then I realized that was 'cheating' because it used information from future games to predict past ones. I fixed this by building a `rolling_strength` score that, for any given game, only uses the team's win percentage from games played *before* that date.
    * **Team Form:** To see if a team was on a hot or cold streak, I added another feature for their win percentage over just their last 5 games.
    * **One-Hot Encoding:** To help the model understand who was playing, I converted the text-based team names into a numeric format it could work with.

* **Honest Model Testing:** A huge lesson was realizing I shouldn't randomize my test data for a sports prediction project. I switched to a **chronological split**, training the model on all past seasons and then testing it only on the most recent season. The accuracy score dropped at first, but it became an *honest* score that wasn't inflated by looking at the future.

* **Model Upgrades:** I started with a simple Logistic Regression model to get a baseline, but then upgraded to a more powerful `XGBoost` model and tuned it to get a better final result.

---

### Final Results

After all the cleaning, feature engineering, and tuning, the final model achieved **57.66% accuracy** on the most recent season of games (the test set).

I then used the final trained model to predict the upcoming week's games. Here are the results:

| Home Team                 | Away Team               | Predicted Winner | Win Probability |
| ------------------------- | ----------------------- | ---------------- | --------------- |
| Pittsburgh Steelers       | Minnesota Vikings       | Steelers         | 65.7%           |
| Atlanta Falcons           | Washington Commanders   | Commanders       | 55.6%           |
| Buffalo Bills             | New Orleans Saints      | Bills            | 59.1%           |
| Detroit Lions             | Cleveland Browns        | Lions            | 50.5%           |
| Houston Texans            | Tennessee Titans        | Titans           | 50.2%           |
| New England Patriots      | Carolina Panthers       | Patriots         | 63.1%           |
| New York Giants           | Los Angeles Chargers    | Chargers         | 57.7%           |
| Tampa Bay Buccaneers      | Philadelphia Eagles     | Eagles           | 55.5%           |
| Los Angeles Rams          | Indianapolis Colts      | Rams             | 54.9%           |
| San Francisco 49ers       | Jacksonville Jaguars    | 49ers            | 64.6%           |
| Kansas City Chiefs        | Baltimore Ravens        | Chiefs           | 60.3%           |
| Las Vegas Raiders         | Chicago Bears           | Raiders          | 54.4%           |
| Dallas Cowboys            | Green Bay Packers       | Cowboys          | 57.4%           |
| Miami Dolphins            | New York Jets           | Jets             | 50.1%           |
| Denver Broncos            | Cincinnati Bengals      | Broncos          | 65.5%           |

---

### Tech Stack

* Python
* pandas (for data manipulation)
* scikit-learn (for splitting data and evaluation)
* XGBoost (for the final prediction model)

---

### How to Run It

1.  Make sure you have Python installed.
2.  Clone this repository to your machine.
3.  Set up a virtual environment: `python -m venv venv` and activate it.
4.  Install the necessary libraries: `pip install -r requirements.txt`.
5.  Place the `spreadspoke_scores.csv` file in the main project folder.
6.  Run the script from your terminal: `python main.py`.