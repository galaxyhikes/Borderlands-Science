######### IMPORT LIBRARIES #########
import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import statsmodels.api as sm
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)  # Show all columns when displaying a DataFrame
pd.set_option('display.max_rows', None)  # Show all rows when displaying a Data
warnings.filterwarnings('ignore')

file_path = '/Users/jiajue/Documents/McGill/Fall Term/Independent study/appending_file_num.csv'

def prep_org(df):
    ''' Prepares the original df for further feature engineering'''
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Drop dups
    df = df.drop_duplicates()

    # Truncate decimal places in par score
    df['par_score_truncated'] = np.floor(df['par_score'])

    # Convert duration col from ms to seconds
    df['duration_sec'] = df['duration'] / 1000

    # Track puzzle seq
    df['puzzle_seq'] = df.sort_values(['player_id', 'difficulty', 'timestamp']).groupby('player_id').cumcount() + 1

    # Create session tracking features for easier reference later on
    df.sort_values(by=['player_id', 'timestamp'], inplace=True)
    df['time_between_plays_min'] = df.groupby('player_id')['timestamp'].diff().dt.total_seconds()/60
    df['time_between_plays_min'].fillna(0, inplace=True) # Fill NaN values for first play with 0
    df['new_session'] = df['time_between_plays_min'] > 15 # 15 min inactivity threshold - align with rob's research paper
    df['session_id'] = df.groupby('player_id')['new_session'].cumsum() # Cumulative sum to create session IDs
    df['puzzle_in_session'] = df.groupby(['player_id', 'session_id']).cumcount() + 1 # Puzzle index within each session

    return (df.drop(columns=['new_session', 'par_score', 'duration'])) # Drop new_session and org par score col as they are no longer needed


def covid_engage_basic(df):

    '''Basic engagement metrics & to create COVID flag'''

    df = df[df['difficulty']!=9]

    df = df.groupby('player_id').agg(
        first_play=('timestamp', 'min'),
        last_play=('timestamp', 'max'),
        covid_flag=('timestamp', lambda x: int(((x >= '2020-03-01') & (x <= '2023-05-31')).all())), # indicator for players who played exclusively only during covid
        active_days=('timestamp', lambda x: x.dt.date.nunique()), # number of unique days the player was active
    ).reset_index()

    # Calculate total days between first and last play
    df['engage_duration'] = (df['last_play'] - df['first_play']).dt.days + 1 # +1 to include both start and end dates
    # engage duration computation method considers time delta and not calendar days that were touched so it is possible for engage duration < active days

    df['activity_ratio'] = df['active_days'] / df['engage_duration']

    # Apply log transformation to reduce skewness
    #df['log_engage_duration'] = np.log(df['engage_duration'])

    # Extract join year and month
    df['join_year'] = df['first_play'].dt.year
    df['join_month'] = df['first_play'].dt.month

    return df.drop(columns=['first_play']) # Drop first play col as it is no longer needed


def puzzle_count(df):
    '''Counts number of puzzles played by each player'''
    # Define puzzle requirements for each level (1-8)
    level_req = {
        1: 10, 2: 15, 3: 20, 4: 25,
        5: 30, 6: 35, 7: 40, 8: 50
    }

    # Define bins for level 9 puzzle count
    bin_edges = [0, 10, 100, 1000, 10000]
    bin_labels = ['1-10 puzzles', '10-100 puzzles', '100-1000 puzzles', '1000+ puzzles']

    levels = df.groupby(['player_id', 'difficulty'])['puzzle_id'].count().reset_index(name='puzzles_completed')

    # Pivot to create cols for each difficulty level
    puzzle_counts = levels.pivot(index='player_id', columns='difficulty', values='puzzles_completed').fillna(0)

    # Rename columns for clarity
    puzzle_counts.columns = [f'level_{col}_puzzles' for col in puzzle_counts.columns]

    # Add total puzzles col - excludes level 9
    puzzle_counts['total_puzzles'] = puzzle_counts[[f'level_{col}_puzzles' for col in level_req]].sum(axis=1)

    # Apply log to level 9 puzzle count
    #puzzle_counts['log_level_9'] = np.log(puzzle_counts['level_9_puzzles'])

    # Use pd.cut to create a new categorical column for level 9 puzzle count
    puzzle_counts['level9_category'] = pd.cut(
        puzzle_counts['level_9_puzzles'],
        bins=bin_edges,
        labels=bin_labels,
        right=True,
        include_lowest=True
    )

    # Calculate proportion of excess puzzles for each level
    for level, req in level_req.items():
        # Compute number of excess puzzles per level
        puzzle_counts[f'level_{level}_excess'] = (puzzle_counts[f'level_{level}_puzzles'] - req).clip(lower=0)

        # proportion of excess puzzles relative to requirement
        puzzle_counts[f'prop_level_{level}_excess'] = puzzle_counts[f'level_{level}_excess'] / req

        # Binary indicator for meeting/exceeding requirement
        puzzle_counts[f'binary_level_{level}_excess'] = (puzzle_counts[f'level_{level}_puzzles'] > req).astype(int)

    # Total number of excess puzzles across all levels
    puzzle_counts['total_excess_puzzles'] = puzzle_counts[[f'level_{level}_excess' for level in level_req]].sum(axis=1)

    # Calculate overall total proportion of puzzles with excess score 
    puzzle_counts['overperf_rate'] = (puzzle_counts['total_excess_puzzles'] / puzzle_counts['total_puzzles']).fillna(0)

    # Drop the individual level excess columns to reduce redundancy
    puzzle_counts.drop(columns=[f'level_{level}_excess' for level in level_req], inplace=True)
    puzzle_counts.drop(columns=[f'level_{level}_puzzles' for level in level_req], inplace=True)

    # Reset index to make player_id a regular column
    puzzle_counts.reset_index(inplace=True)

    return puzzle_counts


def progress_col_engineer(df):
    '''Breaks down progress column into more useful features'''
    # Count total number of clicks from progress column
    df['num_clicks'] = df['progress'].str.count(';') + 1

    # Extract starting score from progress col
    df['starting_score'] = round(df['progress'].str.strip().str.extract(r'^(\d+(?:\.\d+)?)').astype(float)) # ? makes the decimal portion optional

    # Calculate score needed for each puzzle
    df['score_needed_puzzle'] = np.where(
        df['starting_score'] < df['par_score_truncated'],
        df['par_score_truncated'] - df['starting_score'],
        0
    )

    # Calculate excess score achieved beyond par score accounting for final score < par score - assign 0
    df['excess score'] = np.where(
        df['score'] - df['par_score_truncated'] > 0,
        df['score'] - df['par_score_truncated'],
        0
    )

    # Calculate achievement metrics
    ### 1. Excess score using par score as benchmark
    df['excess_ratio'] = np.where(
        df['par_score_truncated'] > 0,
        df['excess score'] / df['par_score_truncated'],
        0
    )

    ### 2. Effort metric using par score as benchmark - effort being the number of points needed to complete from start point
    df['effort_ratio'] = np.where(
        df['score_needed_puzzle'] > 0,
        df['score_needed_puzzle'] / df['par_score_truncated'],
        0
    )

    ### Calculate difficulty level medians for the 2 ratios
    difficulty_medians = df.groupby('difficulty').agg({
        'excess_ratio': 'median',
        'effort_ratio': 'median'
    }).reset_index()

    # Merge medians back to original df
    df = df.merge(difficulty_medians, on='difficulty', suffixes=('', '_median'))

    ### 3. Relative excess and effort ratios compared to median for that difficulty level
    df['relative_effort'] = df['effort_ratio'] / df['effort_ratio_median']
    df['relative_excess'] = df['excess_ratio'] / df['excess_ratio_median']

    # Apply square root transformation on the excess metrics to reduce skewness
    df = df.sort_values(by=['player_id', 'timestamp'])
    df['sqrt_excess_ratio'] = np.sqrt(df['excess_ratio'])
    df['sqrt_relative_excess'] = np.sqrt(df['relative_excess'])

    # Engineer lag features
    df['lag_sqrt_relative_excess'] = df.groupby('player_id')['sqrt_relative_excess'].shift(1)
    df['lag_sqrt_excess_ratio'] = df.groupby('player_id')['sqrt_excess_ratio'].shift(1)

    return df


def effort_excess_corr(player_data):
    '''
    Calculate correlations between effort and excess metrics for each player from levels 1-8. Uses the output from progress_col_engineer function.
    Call using df.groupby('player_id').apply(effort_excess_corr)


    1. Considers behaviours around achieving score > par 
    2. Looks at the r/s between current and lagged performance
    3. Looks at the r/s between effort and achievement
    4. Looks at the r/s between number of puzzles played and achievement
    5. Variation in excess metrics
    6. Streak metrics - longest, avg, total number of streaks where excess score > 0
    7. Level progression effect - difference in excess score between first 25% and last 25% of puzzles in each level
    
    
    # Output for final merging
    '''

    player_data = player_data[player_data['difficulty']!=9].sort_values('timestamp')

    effort_excess_values = {}

    # Current vs lagged performance correlations
    effort_excess_values['consistency_excess'] = player_data['sqrt_excess_ratio'].corr(player_data['lag_sqrt_excess_ratio'])
    effort_excess_values['relative_consistency_excess'] = player_data['sqrt_relative_excess'].corr(player_data['lag_sqrt_relative_excess'])
    
    # Effort vs achievement response
    effort_excess_values['effort_excess_corr'] = player_data['effort_ratio'].corr(player_data['sqrt_excess_ratio'])
    effort_excess_values['relative_effort_excess_corr'] = player_data['relative_effort'].corr(player_data['sqrt_relative_excess'])

    # Excess response to number of puzzles played
    effort_excess_values['puzzles_excess_corr'] = player_data['puzzle_seq'].corr(player_data['sqrt_excess_ratio'])
    effort_excess_values['puzzles_relative_excess_corr'] = player_data['puzzle_seq'].corr(player_data['sqrt_relative_excess'])

    # Variation in the excess metrics
    effort_excess_values['excess_variation'] = player_data['sqrt_excess_ratio'].std()
    effort_excess_values['relative_excess_variation'] = player_data['sqrt_relative_excess'].std()

    # Streak metrics - using excess score as the 'streak' indicator
    above_par = player_data['sqrt_excess_ratio'] > 0 # boolean series indicating if excess score > 0
    streaks = (above_par != above_par.shift()).cumsum() # identify streaks
    streak_lengths = streaks[above_par].value_counts() # lengths of streaks where excess score > 0
    valid_streaks = streak_lengths[streak_lengths >= 2] # consider streaks of length >= 2
    effort_excess_values['longest_excess_streak'] = valid_streaks.max() if not valid_streaks.empty else 0
    effort_excess_values['avg_excess_streak'] = valid_streaks.mean() if not valid_streaks.empty else 0
    effort_excess_values['total_excess_streaks'] = len(valid_streaks) # total number of valid streaks

    # Level progression effect metrics
    level_timing_effect = []
    for level in player_data['difficulty'].unique():
        level_data = player_data[player_data['difficulty'] == level].sort_values('timestamp')
        n_puzzles = len(level_data)
        
        segment_size = int(n_puzzles * 0.25) # Consider the first and last 25% of puzzles in each level
        first_segment = level_data['sqrt_excess_ratio'].iloc[:segment_size].mean()
        last_segment = level_data['sqrt_excess_ratio'].iloc[-segment_size:].mean()
        level_effect = last_segment - first_segment
        level_timing_effect.append(level_effect)

    effort_excess_values['avg_level_progress_effect'] = np.mean(level_timing_effect) if level_timing_effect else 0

    return pd.Series(effort_excess_values)


def find_first_par_time(progress_str, par_score, start_score, end_score, duration, num_clicks):
    '''
    Aims to achieve a few things after locating the first occurrence of reaching/exceeding par score:

    Leading up to reaching par score:
    1. Calculate the time taken to reach that score from the progress column
    2. Calculate the number of clicks taken to reach that score
    3. Calculate the time per click to reach that score
    4. Calculate the proportion of time taken to reach that score against total duration
    5. Calculate the proportion of clicks taken to reach that score against total clicks
    6. Calculate the time per score to reach that score (time taken / (score - par score))
    7. Calculate the clicks per score to reach that score (clicks taken / (score - par score))

    After reaching par score:
    8. Calculate the time taken after reaching that score
    9. Calculate the number of clicks taken after reaching that score
    10. Calculate the time per click after reaching that score
    11. Calculate the proportion of time taken after reaching that score against total duration
    12. Calculate the proportion of clicks taken after reaching that score against total clicks
    13. Calculate the time per score after reaching that score (time taken / (score - par score))
    14. Calculate the clicks per score after reaching that score (clicks taken / (score - par score))

    Returns a tuple of the above 14 metrics
    '''
    if pd.isna(progress_str):
        return None, None

    steps = progress_str.split(';')
    cumulative_time = 0.0
    clicks = 0

    for step in steps:
        score, time = map(float, step.split(':'))
        cumulative_time += time
        clicks += 1
        # Round score to nearest integer before comparing with par score 
        # Done so because the eventual score recorded is in integer form anyway and par score has been truncated
        if round(score) >= par_score:

            # Account for any discrepancies where the sum of times in progress exceeds the recorded duration, cap at duration
            if cumulative_time > duration:
                cumulative_time = duration
            
            ##### Before par metrics
            clicks_to_par = clicks
            time_to_par = cumulative_time
            time_per_click_to_par = cumulative_time / clicks if clicks > 0 else 0
            prop_time_to_par = cumulative_time / duration
            prop_clicks_to_par = clicks / num_clicks if num_clicks > 0 else 0

            # Score based before par metrics
            time_per_score_to_par = cumulative_time / round(score - start_score) if round(score - start_score) > 0 else 0
            clicks_per_score_to_par = clicks / round(score - start_score) if round(score - start_score) > 0 else 0

            ##### After par metrics
            time_after_par = duration - cumulative_time
            clicks_after_par = num_clicks - clicks
            time_per_click_after_par = (duration - cumulative_time) / (num_clicks - clicks) if (num_clicks - clicks) > 0 else 0
            prop_time_after_par = (duration - cumulative_time) / duration
            prop_clicks_after_par = (num_clicks - clicks) / num_clicks if num_clicks > 0 else 0
            
            # Score based after par
            time_per_score_after_par = (duration - cumulative_time) / round(end_score - par_score) if round(end_score - par_score) > 0 else 0
            time_per_click_after_par = (num_clicks - clicks) / round(end_score - par_score) if round(end_score - par_score) > 0 else 0

            # Return all 14 metrics as a tuple
            return (clicks_to_par, time_to_par, time_per_click_to_par, prop_time_to_par, prop_clicks_to_par, time_per_score_to_par, clicks_per_score_to_par,
                    time_after_par, clicks_after_par, time_per_click_after_par, prop_time_after_par, prop_clicks_after_par, time_per_score_after_par, time_per_click_after_par)
            # Calculate clicks, time taken to reach par score, time per click to par, proportion of time and clicks to par score (before par)

    # If par score never reached, return total duration and total clicks but cap time at duration
    # Before par
    return (num_clicks, duration, duration / num_clicks, 1, 1, 
            duration / (score - par_score) if (score - par_score) > 0 else 0, 
            num_clicks / (score - par_score) if (score - par_score) > 0 else 0,
            # After par
            0, 0, 0, 0, 0, 0, 0)

    ### Call using df[['clicks_to_par', 'time_to_par', 'time_per_click_to_par', 'prop_time_to_par', 'prop_clicks_to_par', 'time_per_score_to_par', 'clicks_per_score_to_par',
    #                    'time_after_par', 'clicks_after_par', 'time_per_click_after_par', 'prop_time_after_par', 'prop_clicks_after_par', 'time_per_score_after_par', 'clicks_per_score_after_par']] = df.apply(
    #lambda x: find_first_par_time(x['progress'], 
    #                                x['par_score_truncated'], 
    #                                x['starting score'],
    #                                x['score'],
    #                                x['duration_sec'],
    #                                x['num_clicks']),
    #axis=1, result_type='expand' # Converts returned tuple into separate cols
    #)


def learning_metric(player_data, seq_length=45):
    '''
    Compute learning metrics for at player level using sequence length = 45 as the cut off
    '''
    player_data = player_data[player_data['puzzle_seq'] <= seq_length]

    # Fit learning curves for both metrics
    X = sm.add_constant(player_data['puzzle_seq'])
    y_click = player_data['time_per_click_to_par']
    y_score = player_data['time_per_score_to_par']

    model_click = sm.OLS(y_click, X).fit()
    model_score = sm.OLS(y_score, X).fit()

    return pd.Series({
        'learning_click_slope': model_click.params['puzzle_seq'],
        'r2_learning_click': model_click.rsquared,
        'learning_score_slope': model_score.params['puzzle_seq'],
        'r2_learning_score': model_score.rsquared
    })

    # call using df.groupby('player_id').apply(learning_metric).reset_index(). output for merging

def post_par_analysis(player_data):
    """
    Use click-based metrics to avoid dealing with mostly 0s - time per click after par & prop clicks after par
    Returns pandas Series for easier DataFrame creation
    """
    player_data = player_data[player_data['difficulty']!=9]

    # Return metrics as a pandas Series instead of dictionary
    return pd.Series({
        # Click proportion after par - Engagement metric
        'freq_any_clicks_after_par': (player_data['clicks_after_par'] > 0).mean(), # Exploration frequency
        'avg_prop_clicks_after_par': player_data['prop_clicks_after_par'].mean(), # Exploration extent
        'std_prop_clicks_after_par': player_data['prop_clicks_after_par'].std(),
        'exploration_intensity': (
            (player_data['clicks_after_par'] > 0).mean() * 
            player_data['prop_clicks_after_par'].mean()
        ),
        
        # Time per click after par - Efficiency metric
        'avg_time_per_click_after_par': player_data[
            player_data['time_per_click_after_par'] > 0
        ]['time_per_click_after_par'].mean() if len(player_data[
            player_data['time_per_click_after_par'] > 0
        ]) > 0 else 0,
        
        # Can consider binning into different archetypes based on avg time per click after par

        'std_time_per_click_after_par': player_data[
            player_data['time_per_click_after_par'] > 0
        ]['time_per_click_after_par'].std() if len(player_data[
            player_data['time_per_click_after_par'] > 0
        ]) > 0 else 0
    })

    # Call using df.groupby('player_id').apply(post_par_analysis)


def session_analysis(player_data):
    """
    Compute comprehensive session metrics for each player:
    - Inter-session and inter-puzzle timing - avg time between consecutive sessions, avg time between consecutive puzzles
    - Session counts (short <30min vs long >=30min)  
    - Average session stats (duration, puzzles, performance)
    - Longest session analysis (duration, puzzles, performance change, correlations)
    - Time pattern metrics for longest session
    """
    metrics = {}
    player_data = player_data[player_data['difficulty']!=9]
    
    # Calculate actual session durations using timestamps
    sessions = player_data.groupby('session_id')
    session_times = sessions.agg({
        'timestamp': ['first', 'last']  # Get start and end time of each session
    })
    session_times.columns = ['session_start', 'session_end']

    # Calculate actual session duration in minutes
    session_durations = (session_times['session_end'] - session_times['session_start']).dt.total_seconds() / 60

    # Compute actual gaps between sessions
    session_times = session_times.sort_values('session_start')
    session_times['next_session_start'] = session_times['session_start'].shift(-1)
    session_times['prev_session_end'] = session_times['session_end'].shift(1)
    
    # Time between sessions and puzzles in minutes
    time_between_sessions = (session_times['session_start'] - session_times['prev_session_end']).dt.total_seconds() / 60
    metrics['avg_time_between_sessions_min'] = time_between_sessions.mean() if len(time_between_sessions) > 1 else 0
    metrics['avg_time_between_puzzles_min'] = player_data['time_between_plays_min'].mean()
    
    # Basic session counts and ratios
    short_sessions = len(session_durations[session_durations < 30])
    long_sessions = len(session_durations[session_durations >= 30])
    metrics.update({
        #'num_short_sessions': short_sessions,
        #'num_long_sessions': long_sessions,
        'prop_long_sessions': long_sessions / (short_sessions + long_sessions) if (short_sessions + long_sessions) > 0 else 0
    })
    
    # Average session metrics
    # Get puzzle counts per session separately
    puzzles_per_session = sessions.size()
    
    # Get performance metrics with proper aggregation
    session_performance = sessions.agg({
        'sqrt_excess_ratio': ['mean', 'std'],
        'sqrt_relative_excess': ['mean', 'std']
    })
    
    metrics.update({
        'avg_session_duration': session_durations.mean(),
        'avg_puzzles_per_session': puzzles_per_session.mean(), 
        'avg_session_excess': session_performance['sqrt_excess_ratio']['mean'].mean(),
        'avg_session_relative_excess': session_performance['sqrt_relative_excess']['mean'].mean(),
        'avg_session_excess_std': session_performance['sqrt_excess_ratio']['std'].mean(),
        'avg_session_relative_excess_std': session_performance['sqrt_relative_excess']['std'].mean()
    })
    
    # Intensity metric
    puzzles_per_min = puzzles_per_session / session_durations
    puzzles_per_min = puzzles_per_min.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero if any session duration is 0
    metrics['avg_session_intensity'] = puzzles_per_min.mean()
    metrics['max_session_intensity'] = puzzles_per_min.max()
    metrics['intensity_consistency'] = puzzles_per_min.std() # Variation in session intensity based on puzzle counts

    # Longest session analysis
    if len(session_durations) > 0:
        longest_session_id = session_durations.idxmax()
        longest_session = player_data[player_data['session_id'] == longest_session_id]
        session_duration = session_durations[longest_session_id]
        n_puzzles = len(longest_session)
        
        metrics['longest_session_duration'] = session_duration
        metrics['longest_session_length'] = n_puzzles
        
        if n_puzzles >= 5:
            # Performance metrics for longest session
            third = max(1, n_puzzles // 3)
            
            # Performance decline metrics - excess ratio
            start_raw = longest_session['sqrt_excess_ratio'].iloc[:third].mean()
            end_raw = longest_session['sqrt_excess_ratio'].iloc[-third:].mean()
            raw_decline = (start_raw - end_raw)/start_raw if start_raw != 0 else 0
            
            # Relative excess metrics  
            start_relative = longest_session['sqrt_relative_excess'].iloc[:third].mean()
            end_relative = longest_session['sqrt_relative_excess'].iloc[-third:].mean()
            relative_decline = (start_relative - end_relative)/start_relative if start_relative != 0 else 0
            
            # Score and excess correlations
            raw_corr = longest_session['sqrt_excess_ratio'].corr(longest_session['puzzle_in_session'])
            relative_corr = longest_session['sqrt_relative_excess'].corr(longest_session['puzzle_in_session'])
            
            # Time pattern correlations - only calculate if post-par activity exists
            after_par_click_data = longest_session[
                (longest_session['time_per_click_after_par'] > 0) &
                (longest_session['time_per_click_after_par'].notna()) &
                (longest_session['puzzle_in_session'].notna())
            ]

            after_par_score_data = longest_session[
                (longest_session['time_per_score_after_par'].notna()) &  # Consider removing > 0 condition
                (longest_session['puzzle_in_session'].notna())
            ]

            # Calculate correlations with guaranteed non-NaN results
            if len(after_par_click_data) >= 2:
                time_click_corr = after_par_click_data['time_per_click_after_par'].corr(
                    after_par_click_data['puzzle_in_session']
                )
                time_click_corr = 0 if pd.isna(time_click_corr) else time_click_corr
            else:
                time_click_corr = 0

            if len(after_par_score_data) >= 2:
                time_score_corr = after_par_score_data['time_per_score_after_par'].corr(
                    after_par_score_data['puzzle_in_session']
                )
                time_score_corr = 0 if pd.isna(time_score_corr) else time_score_corr
            else:
                time_score_corr = 0
            
            # Time metrics for longest session
            time_metrics = longest_session.agg({
                'time_per_click_after_par': ['mean', 'std'],
                'time_per_score_after_par': ['mean', 'std']
            }).fillna(0)
            
            metrics.update({
                'longest_session_raw_decline': raw_decline,
                'longest_session_relative_decline': relative_decline,
                'longest_session_raw_corr': raw_corr if not pd.isna(raw_corr) else 0,
                'longest_session_relative_corr': relative_corr if not pd.isna(relative_corr) else 0,
                'longest_session_time_click_corr': time_click_corr,
                'longest_session_time_score_corr': time_score_corr,
                'longest_session_avg_time_per_click': time_metrics['time_per_click_after_par']['mean'],
                'longest_session_std_time_per_click': time_metrics['time_per_click_after_par']['std'],
                'longest_session_avg_time_per_score': time_metrics['time_per_score_after_par']['mean'],
                'longest_session_std_time_per_score': time_metrics['time_per_score_after_par']['std']
            })
        else:
            # Fill with zeros if session is too short
            metrics.update({
                'longest_session_raw_decline': 0,
                'longest_session_relative_decline': 0, 
                'longest_session_raw_corr': 0,
                'longest_session_relative_corr': 0,
                'longest_session_time_click_corr': 0,
                'longest_session_time_score_corr': 0,
                'longest_session_avg_time_per_click': 0,
                'longest_session_std_time_per_click': 0,
                'longest_session_avg_time_per_score': 0,
                'longest_session_std_time_per_score': 0
            })
    else:
        # Handle case with no sessions
        metrics.update({
            'longest_session_duration': 0,
            'longest_session_length': 0,
            'longest_session_raw_decline': 0,
            'longest_session_relative_decline': 0, 
            'longest_session_raw_corr': 0,
            'longest_session_relative_corr': 0,
            'longest_session_time_click_corr': 0,
            'longest_session_time_score_corr': 0,
            'longest_session_avg_time_per_click': 0,
            'longest_session_std_time_per_click': 0,
            'longest_session_avg_time_per_score': 0,
            'longest_session_std_time_per_score': 0
        })
    
    return pd.Series(metrics)

    # Call function using df.groupby('player_id').apply(session_analysis)


def transformation_pipeline(df, config=None):
    """
    Complete preprocessing with multiple transformation options
    """
    if config is None:
        config = {
            'skew_threshold': 0.5,
            'encode_dates': True,
            'transform_method': 'auto'  # 'log', 'minmax', 'standard', 'auto'
        }
    
    df_processed = df.copy()
    transformations = []

    # Date encoding
    if config['encode_dates']:
        if 'join_month' in df_processed.columns:
            month_dummies = pd.get_dummies(df_processed['join_month'], prefix='month', drop_first=True)
            df_processed = pd.concat([df_processed, month_dummies], axis=1)
            df_processed = df_processed.drop('join_month', axis=1)
            #transformations['month_encoding'] = 'dummy_variables'
            transformations.append({
                'column': 'join_month',
                'transformation_type': 'encoding',
                'method': 'dummy',
                'original_skew': None,
                'new_skew': None
                #'details': f'Created {month_dummies.shape[1]} dummy variables'
            })
        
        if 'join_year' in df_processed.columns:
            df_processed['years_since_2020'] = df_processed['join_year'] - 2020
            df_processed = df_processed.drop('join_year', axis=1)
            #transformations['year_encoding'] = 'years_since_2020'
            transformations.append({
                'column': 'years_since_2020',
                'transformation_type': 'numeric encoding',
                'method': 'years_since_2020',
                'original_skew': None,
                'new_skew': None
                #'details': 'Converted to years since 2020'
            })

    # Numeric transformations
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

    # Identify cols that are already in appropriate scales and should not be transformed
    corr_cols = [col for col in df_processed.columns if 'corr' in col or 'r2' in col]
    binary_cols = [col for col in df_processed.columns if 'binary' in col]
    prop_cols = [col for col in df_processed.columns if 'prop' in col or 'ratio' in col or '_decline' in col] + \
                ['overperf_rate', 'exploration_intensity']

    skip_cols = ['player_id', 'covid_flag'] + \
                [col for col in df_processed.columns if col.startswith('month') or col.startswith('years_') or col.startswith('level_9_puzzle')] + \
                corr_cols + binary_cols + prop_cols
    
    for col in [c for c in numeric_cols if c not in skip_cols]:
        print('Processing column:', col)
        data = df_processed[col].dropna()
        
        if len(data) < 2 or data.nunique() <= 1:
            transformations.append({
                'column': col,
                'transformation_type': 'skipped',
                'method': 'insufficient_data',
                'original_skew': None,
                'new_skew': None
                #'details': 'Insufficient data for transformation'
            })
            continue
            
        if pd.api.types.is_categorical_dtype(df_processed[col]):
            transformations.append({
                'column': col,
                'transformation_type': 'skipped',
                'method': 'categorical_dtype',
                'original_skew': None,
                'new_skew': None
                #'details': 'Column is categorical type'
            })
            continue

        col_skew = skew(data)
        #print(f"{col}: skewness = {col_skew:.3f}")
        if abs(col_skew) > config['skew_threshold']:
            
            if config['transform_method'] == 'auto':
                # Auto-select best transformation
                transformed, method_used = auto_select_transformation(data, col_skew)
            elif config['transform_method'] == 'minmax':
                transformed = MinMaxScaler().fit_transform(data.values.reshape(-1, 1)).flatten()
                method_used = 'minmax'
            elif config['transform_method'] == 'standard':
                transformed = StandardScaler().fit_transform(data.values.reshape(-1, 1)).flatten()
                method_used = 'standard'
            else:  # log
                if data.min() >= 0:
                    transformed = np.log1p(data)
                    method_used = 'log1p'
                else:
                    transformed = data
                    method_used = 'none'

            df_processed[col] = df_processed[col].astype(float)
            df_processed.loc[data.index, col] = transformed

            new_skew = skew(transformed) if len(transformed) > 2 else None
            transformations.append({
                'column': col,
                'transformation_type': 'numeric',
                'method': method_used,
                'original_skew': col_skew,
                'new_skew': new_skew
            })


    # Convert bool type to int
    bool_cols = df_processed.select_dtypes(include=['bool']).columns
    df_processed[bool_cols] = df_processed[bool_cols].astype(int)

    transformations_df = pd.DataFrame(transformations)

    return df_processed, transformations_df


def auto_select_transformation(data, original_skew):
    """Select transformation based on data characteristics & skewness"""
    min_val, max_val = data.min(), data.max()
    #print(min_val, max_val)
    
    # If non-negative & highly skewed (positive) → log
    if min_val >= 0 and original_skew > 1: # only apply log for highly skewed data to avoid over-compression of moderate skewness
        return np.log1p(data), 'log1p_auto'
    
    # Moderate positive skewness (for non negative data) or bounded → min-max
    elif (min_val >= 0 and original_skew > 0.5) or (min_val >= 0 and max_val <= 1):
        return MinMaxScaler().fit_transform(data.values.reshape(-1, 1)).flatten(), 'minmax_auto'
    
    # Has negative values or negative skew → standardise as a safe approach
    elif min_val < 0 or original_skew < -0.5:
        return StandardScaler().fit_transform(data.values.reshape(-1, 1)).flatten(), 'standard_auto'

    # Default → no transformation
    else:
        return data, 'none_auto'


def survival_censor(processed_df, grace=2.3):
    '''
    Determine censoring status based on player activity patterns

    Allowing 2.3x grace period of avg time between sessions before considering a player to have churned
    '''

    # Ensure last_play is in datetime format
    processed_df['last_play'] = pd.to_datetime(processed_df['last_play'])

    # Latest date in puzzle dataset
    latest_date = processed_df['last_play'].max()
    print(f"Latest play date in dataset: {latest_date}")

    # Calculate time since last play in days
    processed_df['days_since_last_play'] = (latest_date - processed_df['last_play']).dt.days

    # Apply censoring rule - if days since last play < avg time between sessions, censored i.e. assume to still be active
    # if days since last play >= avg time between sessions, not censored i.e. assume to have churned
    processed_df['event'] = (processed_df['days_since_last_play'] >= processed_df['avg_time_between_sessions_min']*grace / 1440).astype(int)

    return processed_df


if __name__ == "__main__":
    try:
        print('Starting player profile generation...')

        # Read file
        df = pd.read_csv(file_path)
        print('File read successfully.')

        # Basic preprocessing
        print('Preprocessing data...')
        df = prep_org(df)
        print('Preprocessing completed.')

        ############ FEATURE ENGINEERING ############
        # COVID flag and engagement metrics
        print('Computing COVID and engagement metrics...')
        player_covid = covid_engage_basic(df)
        print('COVID and engagement metrics computed.')
        
        # Puzzle count
        print('Computing puzzle counts...')
        puzzle_counts = puzzle_count(df)
        print('Puzzle counts computed.')

        # Progress engineer
        print('Extracting features from progress column...')
        progress_breakdown_df = progress_col_engineer(df)

        # Effort-excess correlations
        print('Computing effort-excess correlations...')
        player_effort_excess = progress_breakdown_df.groupby('player_id').apply(effort_excess_corr).reset_index()
        print('Effort-excess correlations computed.')

        # First par time analysis
        print('Computing metrics on reaching par for the first time...')
        progress_breakdown_df[['clicks_to_par', 'time_to_par', 'time_per_click_to_par', 'prop_time_to_par', 'prop_clicks_to_par', 'time_per_score_to_par', 'clicks_per_score_to_par',
                                'time_after_par', 'clicks_after_par', 'time_per_click_after_par', 'prop_time_after_par', 'prop_clicks_after_par', 'time_per_score_after_par', 'clicks_per_score_after_par']] = progress_breakdown_df.apply(
                                    lambda x: find_first_par_time(x['progress'],
                                                                    x['par_score_truncated'],
                                                                    x['starting_score'],
                                                                    x['score'],
                                                                    x['duration_sec'],
                                                                    x['num_clicks']),
                                    axis=1, result_type='expand' # Converts returned tuple into separate cols
                                )
        print('Metrics on reaching par computed.')

        # Learning metrics
        print('Computing learning metrics...')
        player_learning = progress_breakdown_df.groupby('player_id').apply(learning_metric).reset_index()
        print('Learning metrics computed.')

        # Post par analysis
        print('Computing post-par analysis metrics...')
        player_post_par = progress_breakdown_df.groupby('player_id').apply(post_par_analysis).reset_index()
        print('Post-par analysis metrics computed.')
 
        # Session analysis
        print('Computing session analysis metrics...')
        player_session = progress_breakdown_df.groupby('player_id').apply(session_analysis).reset_index()
        print('Session analysis metrics computed.')

        # Merge all player-level dataframes
        player_profile = player_covid.merge(puzzle_counts, on='player_id', how='left') \
                                    .merge(player_effort_excess, on='player_id', how='left') \
                                    .merge(player_learning, on='player_id', how='left') \
                                    .merge(player_post_par, on='player_id', how='left') \
                                    .merge(player_session, on='player_id', how='left')

        print('All player-level metrics merged.')

        #print(player_profile.info())
        #print(player_profile.describe()) 

        # Fill in any missing values
        player_profile = player_profile.apply(
            lambda col: col.fillna(0) if col.dtype != 'category' else col
        )

        # Fail save
        player_profile.to_csv('player_profile.csv', index=False)

        # Apply transformations
        player_profile = pd.read_csv('/Users/jiajue/Documents/McGill/Fall Term/Independent study/player_profile.csv')
        print('Applying transformations...')
        df_process, transform_info = transformation_pipeline(player_profile)
        print('Transformations done.')

        # Apply survival censoring
        print('Applying survival censoring...')
        df_process = survival_censor(df_process)
        print('Survival censoring done.')

        # Final save
        df_process.to_csv('player_profile_processed.csv', index=False)
        transform_info.to_csv('transformation_log.csv', index=False)

        print('Player profile generation completed successfully.')

    except Exception as e:
        print(f"Error occurred: {e}")