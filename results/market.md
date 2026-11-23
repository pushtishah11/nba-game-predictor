Walk-forward comparison with the betting market on 11,741 games (2017-18 to now). Market probability = median no-vig probability across sportsbooks' final pre-game moneylines.

| | Accuracy | Log loss | Brier |
|---|---|---|---|
| Betting market | 67.8% | 0.5983 | 0.2060 |
| My model | 66.3% | 0.6126 | 0.2123 |
| My model, no lineups | 65.3% | 0.6220 | 0.2165 |
| Elo only | 64.6% | 0.6336 | 0.2212 |

The model and the market pick different winners in 1,250 games (10.6%); the model is right in 43.4% of those.

**Market + model blend** (fit on earlier seasons, tested on the next):

| | Accuracy | Log loss | Brier |
|---|---|---|---|
| Market | 67.7% | 0.5992 | 0.2064 |
| Market + my model | 67.6% | 0.5993 | 0.2064 |

Average weight on the model: -11%.

**Betting simulation** (1 unit per bet at the median moneyline):

| Edge needed | Bets | Win rate | Profit (units) | ROI |
|---|---|---|---|---|
| 0% | 8,771 | 40.5% | -432.7 | -4.9% |
| 3% | 7,073 | 37.4% | -421.2 | -6.0% |
| 5% | 6,187 | 35.7% | -404.5 | -6.5% |
| 10% | 4,368 | 31.4% | -356.2 | -8.2% |
