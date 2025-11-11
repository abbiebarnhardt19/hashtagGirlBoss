from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Import your simulator class here (copy the entire BaseballGameSimulator class)
# Or if it's in a separate file: from your_simulator import BaseballGameSimulator
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GameState:
    """Track the current state of the game"""
    inning: int = 1
    outs: int = 0
    bases: List[bool] = None  # [1st, 2nd, 3rd]
    home_score: int = 0
    away_score: int = 0
    
    def __post_init__(self):
        if self.bases is None:
            self.bases = [False, False, False]
    
    def clear_bases(self):
        self.bases = [False, False, False]
    
    def advance_runners(self, bases_advanced: int):
        """Advance runners and return runs scored"""
        runs = 0
        # Go backwards through bases to avoid double-counting
        for i in range(2, -1, -1):
            if self.bases[i]:
                new_base = i + bases_advanced
                if new_base >= 3:
                    runs += 1
                    self.bases[i] = False
                else:
                    self.bases[new_base] = True
                    self.bases[i] = False
        return runs

@dataclass
class PlayerGameStats:
    """Track individual player stats during simulation"""
    player_name: str
    at_bats: int = 0
    hits: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    strikeouts: int = 0
    rbis: int = 0
    runs: int = 0

@dataclass
class PitcherGameStats:
    """Track pitcher stats during simulation"""
    pitcher_name: str
    strikeouts: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    runs_allowed: int = 0
    pitches_thrown: int = 0

class BaseballGameSimulator:
    def __init__(self, pitcher_df: pd.DataFrame, batter_df: pd.DataFrame):
        self.pitcher_df = pitcher_df
        self.batter_df = batter_df
        
    def get_pitcher_stats(self, pitcher_name: str) -> Dict:
        """Extract relevant pitcher stats by name"""
        pitcher_matches = self.pitcher_df.loc[self.pitcher_df['PlayerName'] == pitcher_name]
        
        if len(pitcher_matches) == 0:
            raise ValueError(f"Pitcher '{pitcher_name}' not found in pitcher dataframe")
        
        pitcher = pitcher_matches.iloc[0]
        
        # Calculate per-batter-faced probabilities
        bf = pitcher['battersFaced']
        if bf == 0:
            bf = 1  # Avoid division by zero
            
        return {
            'name': pitcher['PlayerName'],
            'k_rate': pitcher['strikeOuts'] / bf,
            'bb_rate': pitcher['baseOnBalls'] / bf,
            'hit_rate': pitcher['hits'] / bf,
            'hr_rate': pitcher['homeRuns'] / bf,
            'double_rate': pitcher['doubles'] / bf,
            'triple_rate': pitcher['triples'] / bf,
        }
    
    def get_batter_stats(self, batter_name: str) -> Dict:
        """Extract relevant batter stats by name"""
        batter_matches = self.batter_df.loc[self.batter_df['PlayerName'] == batter_name]
        
        if len(batter_matches) == 0:
            raise ValueError(f"Batter '{batter_name}' not found in batter dataframe")
        
        batter = batter_matches.iloc[0]
        
        # Calculate probabilities
        pa = batter['plateAppearances']
        if pa == 0:
            pa = 1
            
        return {
            'name': batter['PlayerName'],
            'k_rate': batter['strikeOuts'] / pa,
            'bb_rate': batter['baseOnBalls'] / pa,
            'hit_rate': batter['hits'] / pa,
            'hr_rate': batter['homeRuns'] / pa,
            'double_rate': batter['doubles'] / pa,
            'triple_rate': batter['triples'] / pa,
            'single_rate': batter.get('singles', 0) / pa if 'singles' in batter else 0,
        }
    
    def simulate_plate_appearance(self, batter_stats: Dict, pitcher_stats: Dict) -> Tuple[str, int]:
        """
        Simulate a single plate appearance
        Returns: (outcome, bases_advanced)
        Outcomes: 'K', 'BB', 'single', 'double', 'triple', 'HR', 'out'
        """
        # Blend batter and pitcher stats (60% batter, 40% pitcher weight)
        k_prob = 0.6 * batter_stats['k_rate'] + 0.4 * pitcher_stats['k_rate']
        bb_prob = 0.6 * batter_stats['bb_rate'] + 0.4 * pitcher_stats['bb_rate']
        hr_prob = 0.6 * batter_stats['hr_rate'] + 0.4 * pitcher_stats['hr_rate']
        double_prob = 0.6 * batter_stats['double_rate'] + 0.4 * pitcher_stats['double_rate']
        triple_prob = 0.6 * batter_stats['triple_rate'] + 0.4 * pitcher_stats['triple_rate']
        
        # Single rate (hits minus extra base hits)
        single_prob = (0.6 * batter_stats['hit_rate'] + 0.4 * pitcher_stats['hit_rate']) - \
                      (hr_prob + double_prob + triple_prob)
        single_prob = max(0, single_prob)  # Can't be negative
        
        # Remaining probability is outs (includes ground outs, fly outs, etc.)
        total_prob = k_prob + bb_prob + hr_prob + double_prob + triple_prob + single_prob
        out_prob = max(0, 1 - total_prob)
        
        # Simulate outcome
        rand = np.random.random()
        cumulative = 0
        
        cumulative += k_prob
        if rand < cumulative:
            return 'K', 0
        
        cumulative += bb_prob
        if rand < cumulative:
            return 'BB', 1
        
        cumulative += hr_prob
        if rand < cumulative:
            return 'HR', 4
        
        cumulative += triple_prob
        if rand < cumulative:
            return 'triple', 3
        
        cumulative += double_prob
        if rand < cumulative:
            return 'double', 2
        
        cumulative += single_prob
        if rand < cumulative:
            return 'single', 1
        
        return 'out', 0
    
    def simulate_half_inning(self, batting_lineup: List[str], pitcher_name: str, 
                            batter_idx: int, game_state: GameState, 
                            player_stats: Dict[str, PlayerGameStats],
                            pitcher_stats_tracker: Dict[str, PitcherGameStats]) -> Tuple[int, int]:
        """
        Simulate a half inning
        Returns: (runs_scored, next_batter_idx)
        """
        runs = 0
        outs = 0
        game_state.clear_bases()
        
        pitcher_stats = self.get_pitcher_stats(pitcher_name)
        
        # Initialize pitcher stats tracker if needed
        if pitcher_name not in pitcher_stats_tracker:
            pitcher_stats_tracker[pitcher_name] = PitcherGameStats(pitcher_name=pitcher_name)
        
        while outs < 3:
            batter_name = batting_lineup[batter_idx]
            batter_stats = self.get_batter_stats(batter_name)
            
            # Get or create player stats
            if batter_name not in player_stats:
                player_stats[batter_name] = PlayerGameStats(
                    player_name=batter_name
                )
            
            outcome, bases_advanced = self.simulate_plate_appearance(batter_stats, pitcher_stats)
            
            # Track pitcher stats
            pitcher_stats_tracker[pitcher_name].pitches_thrown += 1
            
            if outcome == 'K':
                outs += 1
                player_stats[batter_name].at_bats += 1
                player_stats[batter_name].strikeouts += 1
                pitcher_stats_tracker[pitcher_name].strikeouts += 1
                
            elif outcome == 'out':
                outs += 1
                player_stats[batter_name].at_bats += 1
                
            elif outcome == 'BB':
                player_stats[batter_name].walks += 1
                pitcher_stats_tracker[pitcher_name].walks_allowed += 1
                # Walk advances runners if bases loaded
                if all(game_state.bases):
                    runs += 1
                    player_stats[batter_name].rbis += 1
                else:
                    # Walk with force
                    if game_state.bases[0]:
                        if game_state.bases[1]:
                            if game_state.bases[2]:
                                runs += 1
                                player_stats[batter_name].rbis += 1
                            else:
                                game_state.bases[2] = True
                        else:
                            game_state.bases[1] = True
                    game_state.bases[0] = True
                    
            elif outcome == 'HR':
                player_stats[batter_name].at_bats += 1
                player_stats[batter_name].hits += 1
                player_stats[batter_name].home_runs += 1
                pitcher_stats_tracker[pitcher_name].hits_allowed += 1
                # Count runners on base
                runners_on = sum(game_state.bases)
                runs += runners_on + 1  # Runners plus batter
                player_stats[batter_name].rbis += runners_on + 1
                player_stats[batter_name].runs += 1
                game_state.clear_bases()
                
            elif outcome in ['single', 'double', 'triple']:
                player_stats[batter_name].at_bats += 1
                player_stats[batter_name].hits += 1
                pitcher_stats_tracker[pitcher_name].hits_allowed += 1
                
                if outcome == 'single':
                    player_stats[batter_name].singles += 1
                elif outcome == 'double':
                    player_stats[batter_name].doubles += 1
                else:
                    player_stats[batter_name].triples += 1
                
                # Advance runners
                runs_scored = game_state.advance_runners(bases_advanced)
                runs += runs_scored
                player_stats[batter_name].rbis += runs_scored
                
                # Place batter on base
                game_state.bases[bases_advanced - 1] = True
            
            # Move to next batter
            batter_idx = (batter_idx + 1) % len(batting_lineup)
        
        # Track runs allowed by pitcher
        pitcher_stats_tracker[pitcher_name].runs_allowed += runs
        
        return runs, batter_idx
    
    def simulate_game(self, away_lineup: List[str], away_pitcher: str,
                     home_lineup: List[str], home_pitcher: str,
                     innings: int = 9) -> Dict:
        """
        Simulate a complete game
        Returns dictionary with game results
        """
        game_state = GameState()
        away_batter_idx = 0
        home_batter_idx = 0
        
        # Track player stats
        player_stats = {}
        pitcher_stats_tracker = {}
        
        inning_scores = {'away': [], 'home': []}
        
        # Simulate regulation innings
        for inning in range(1, innings + 1):
            # Away team bats (top of inning)
            runs, away_batter_idx = self.simulate_half_inning(
                away_lineup, home_pitcher, away_batter_idx, game_state, player_stats, pitcher_stats_tracker
            )
            game_state.away_score += runs
            inning_scores['away'].append(runs)
            
            # Home team bats (bottom of inning)
            # Don't bat in bottom of 9th if already winning
            if inning == innings and game_state.home_score > game_state.away_score:
                inning_scores['home'].append(0)
                break
                
            runs, home_batter_idx = self.simulate_half_inning(
                home_lineup, away_pitcher, home_batter_idx, game_state, player_stats, pitcher_stats_tracker
            )
            game_state.home_score += runs
            inning_scores['home'].append(runs)
            
            # Check for walk-off
            if inning >= innings and game_state.home_score > game_state.away_score:
                break
        
        # Extra innings if tied
        extra_inning = innings + 1
        while game_state.away_score == game_state.home_score:
            # Away team
            runs, away_batter_idx = self.simulate_half_inning(
                away_lineup, home_pitcher, away_batter_idx, game_state, player_stats, pitcher_stats_tracker
            )
            game_state.away_score += runs
            inning_scores['away'].append(runs)
            
            # Home team
            runs, home_batter_idx = self.simulate_half_inning(
                home_lineup, away_pitcher, home_batter_idx, game_state, player_stats, pitcher_stats_tracker
            )
            game_state.home_score += runs
            inning_scores['home'].append(runs)
            
            if game_state.home_score > game_state.away_score:
                break
                
            extra_inning += 1
            if extra_inning > 15:  # Safety limit
                break
        
        # Calculate total strikeouts
        total_strikeouts = sum(ps.strikeouts for ps in player_stats.values())
        
        return {
            'away_score': game_state.away_score,
            'home_score': game_state.home_score,
            'winner': 'home' if game_state.home_score > game_state.away_score else 'away',
            'total_strikeouts': total_strikeouts,
            'inning_scores': inning_scores,
            'player_stats': player_stats,
            'pitcher_stats': pitcher_stats_tracker,
            'innings_played': len(inning_scores['away'])
        }
    
    def simulate_multiple_games(self, away_lineup: List[str], away_pitcher: str,
                               home_lineup: List[str], home_pitcher: str,
                               n_simulations: int = 1000) -> Dict:
        """
        Simulate multiple games and aggregate results
        """
        results = {
            'away_wins': 0,
            'home_wins': 0,
            'away_scores': [],
            'home_scores': [],
            'total_strikeouts': [],
            'player_hit_frequency': defaultdict(int),
            'player_avg_stats': defaultdict(lambda: {
                'at_bats': 0, 'hits': 0, 'hr': 0, 'rbi': 0, 'k': 0
            }),
            'pitcher_avg_stats': defaultdict(lambda: {
                'strikeouts': 0, 'hits_allowed': 0, 'walks_allowed': 0, 'runs_allowed': 0
            })
        }
        
        for _ in range(n_simulations):
            game_result = self.simulate_game(away_lineup, away_pitcher, 
                                            home_lineup, home_pitcher)
            
            # Aggregate results
            results['away_scores'].append(game_result['away_score'])
            results['home_scores'].append(game_result['home_score'])
            results['total_strikeouts'].append(game_result['total_strikeouts'])
            
            if game_result['winner'] == 'home':
                results['home_wins'] += 1
            else:
                results['away_wins'] += 1
            
            # Track player stats
            for player_name, stats in game_result['player_stats'].items():
                if stats.hits > 0:
                    results['player_hit_frequency'][player_name] += 1
                
                results['player_avg_stats'][player_name]['at_bats'] += stats.at_bats
                results['player_avg_stats'][player_name]['hits'] += stats.hits
                results['player_avg_stats'][player_name]['hr'] += stats.home_runs
                results['player_avg_stats'][player_name]['rbi'] += stats.rbis
                results['player_avg_stats'][player_name]['k'] += stats.strikeouts
            
            # Track pitcher stats
            for pitcher_name, stats in game_result['pitcher_stats'].items():
                results['pitcher_avg_stats'][pitcher_name]['strikeouts'] += stats.strikeouts
                results['pitcher_avg_stats'][pitcher_name]['hits_allowed'] += stats.hits_allowed
                results['pitcher_avg_stats'][pitcher_name]['walks_allowed'] += stats.walks_allowed
                results['pitcher_avg_stats'][pitcher_name]['runs_allowed'] += stats.runs_allowed
        
        # Calculate averages
        results['away_win_pct'] = results['away_wins'] / n_simulations
        results['home_win_pct'] = results['home_wins'] / n_simulations
        results['avg_away_score'] = np.mean(results['away_scores'])
        results['avg_home_score'] = np.mean(results['home_scores'])
        results['avg_total_strikeouts'] = np.mean(results['total_strikeouts'])
        
        # Convert player stats to averages
        for player_name in results['player_avg_stats']:
            for stat in results['player_avg_stats'][player_name]:
                results['player_avg_stats'][player_name][stat] /= n_simulations
        
        # Convert pitcher stats to averages
        for pitcher_name in results['pitcher_avg_stats']:
            for stat in results['pitcher_avg_stats'][pitcher_name]:
                results['pitcher_avg_stats'][pitcher_name][stat] /= n_simulations
        
        return results

app = Flask(__name__)
CORS(app)  # This allows your HTML page to call the API

# Load your data when the server starts
print("Loading data...")
pitcher_data = pd.read_csv('/Users/abigailbarnhardt/Desktop/Trendy Headaches/hashtagGirlBoss/Data/API Output-Pitchers.csv')
batter_data = pd.read_csv('/Users/abigailbarnhardt/Desktop/Trendy Headaches/hashtagGirlBoss/Data/API Output-Batters.csv')
simulator = BaseballGameSimulator(pitcher_data, batter_data)
print("Data loaded successfully!")

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract parameters
        away_lineup = data['away_lineup']
        away_pitcher = data['away_pitcher']
        home_lineup = data['home_lineup']
        home_pitcher = data['home_pitcher']
        n_simulations = data.get('n_simulations', 1000)  # Default to 1000 if not provided
        
        print(f"Running {n_simulations} simulations...")
        print(f"Away: {away_pitcher} vs Home: {home_pitcher}")
        
        # Run simulation
        results = simulator.simulate_multiple_games(
            away_lineup=away_lineup,
            away_pitcher=away_pitcher,
            home_lineup=home_lineup,
            home_pitcher=home_pitcher,
            n_simulations=n_simulations
        )
        
        # Convert results to JSON-serializable format
        response = {
            'success': True,
            'home_win_pct': float(results['home_win_pct']),
            'away_win_pct': float(results['away_win_pct']),
            'avg_home_score': float(results['avg_home_score']),
            'avg_away_score': float(results['avg_away_score']),
            'avg_total_strikeouts': float(results['avg_total_strikeouts']),
            'pitcher_stats': {},
            'player_stats': {},
            'player_hit_frequency': {}
        }
        
        # Convert pitcher stats
        for pitcher_name, stats in results['pitcher_avg_stats'].items():
            response['pitcher_stats'][pitcher_name] = {
                'strikeouts': float(stats['strikeouts']),
                'hits_allowed': float(stats['hits_allowed']),
                'walks_allowed': float(stats['walks_allowed']),
                'runs_allowed': float(stats['runs_allowed'])
            }
        
        # Convert player stats
        for player_name, stats in results['player_avg_stats'].items():
            response['player_stats'][player_name] = {
                'at_bats': float(stats['at_bats']),
                'hits': float(stats['hits']),
                'hr': float(stats['hr']),
                'rbi': float(stats['rbi']),
                'k': float(stats['k'])
            }
        
        # Convert hit frequency
        for player_name, frequency in results['player_hit_frequency'].items():
            response['player_hit_frequency'][player_name] = int(frequency)
        
        print("Simulation completed successfully!")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001, host='0.0.0.0')