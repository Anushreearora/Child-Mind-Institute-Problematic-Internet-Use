import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def add(col1, col2, new_col_name=None):
    """
    Add two columns together
    
    Parameters:
    col1, col2 (pd.Series): Columns to add
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with addition result
    """
    result = col1 + col2
    if new_col_name is not None:
        result.name = new_col_name
    return result

def subtract(col1, col2, new_col_name=None):
    """
    Subtract col2 from col1
    
    Parameters:
    col1, col2 (pd.Series): Columns for subtraction (col1 - col2)
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with subtraction result
    """
    result = col1 - col2
    if new_col_name is not None:
        result.name = new_col_name
    return result

def multiply(col1, col2, new_col_name=None):
    """
    Multiply two columns together
    
    Parameters:
    col1, col2 (pd.Series): Columns to multiply
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with multiplication result
    """
    result = col1 * col2
    if new_col_name is not None:
        result.name = new_col_name
    return result

def divide(col1, col2, new_col_name=None, fillna=0):
    """
    Divide col1 by col2
    
    Parameters:
    col1, col2 (pd.Series): Columns for division (col1 / col2)
    new_col_name (str, optional): Name for the new column
    fillna (float, optional): Value to fill NaN results with (from division by zero)
    
    Returns:
    pd.Series: Series with division result
    """
    result = col1 / col2.replace(0, np.nan)
    result = result.fillna(fillna)
    if new_col_name is not None:
        result.name = new_col_name
    return result

def power(col, power_value, new_col_name=None):
    """
    Raise column values to a power
    
    Parameters:
    col (pd.Series): Column to transform
    power_value (float): Power to raise values to
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with power result
    """
    result = col ** power_value
    if new_col_name is not None:
        result.name = new_col_name
    return result

def abs_val(col, new_col_name=None):
    """
    Calculate absolute value of a column
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with absolute values
    """
    result = col.abs()
    if new_col_name is not None:
        result.name = new_col_name
    return result

def log(col, new_col_name=None, base=np.e, offset=0):
    """
    Calculate log of a column
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    base (float, optional): Log base. Default is natural log (e)
    offset (float, optional): Value to add before taking log (for handling zeros/negatives)
    
    Returns:
    pd.Series: Series with log transformation
    """
    if offset > 0:
        result = np.log(col + offset) / np.log(base)
    else:
        # Handle negative or zero values
        valid_values = col > 0
        result = pd.Series(np.nan, index=col.index)
        result.loc[valid_values] = np.log(col.loc[valid_values]) / np.log(base)
    
    if new_col_name is not None:
        result.name = new_col_name
    return result

def log_diff(col, periods=1, new_col_name=None):
    """
    Calculate log difference (percentage change)
    
    Parameters:
    col (pd.Series): Column to transform
    periods (int, optional): Number of periods to shift for difference
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with log difference
    """
    result = np.log(col) - np.log(col.shift(periods))
    if new_col_name is not None:
        result.name = new_col_name
    return result

def sqrt(col, new_col_name=None):
    """
    Calculate square root of a column
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with square root values
    """
    # Handle negative values
    result = np.sqrt(col.clip(lower=0))
    if new_col_name is not None:
        result.name = new_col_name
    return result

def rank(col, new_col_name=None, method='average'):
    """
    Calculate the rank of values in a column
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    method (str, optional): Method for rank ties ('average', 'min', 'max', 'first', 'dense')
    
    Returns:
    pd.Series: Series with rank values
    """
    result = col.rank(method=method)
    if new_col_name is not None:
        result.name = new_col_name
    return result

def quantile(col, new_col_name=None, q=4):
    """
    Bin values into quantiles
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    q (int, optional): Number of quantiles. Default is 4 (quartiles)
    
    Returns:
    pd.Series: Series with quantile bins
    """
    result = pd.qcut(col, q=q, labels=False, duplicates='drop')
    if new_col_name is not None:
        result.name = new_col_name
    return result

def normalise(col, new_col_name=None, method='zscore'):
    """
    Normalize a column
    
    Parameters:
    col (pd.Series): Column to transform
    new_col_name (str, optional): Name for the new column
    method (str, optional): Normalization method ('zscore', 'minmax')
    
    Returns:
    pd.Series: Series with normalized values
    """
    if method == 'zscore':
        mean = col.mean()
        std = col.std()
        if std == 0:
            result = pd.Series(0, index=col.index)
        else:
            result = (col - mean) / std
    elif method == 'minmax':
        min_val = col.min()
        max_val = col.max()
        if max_val == min_val:
            result = pd.Series(0, index=col.index)
        else:
            result = (col - min_val) / (max_val - min_val)
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")
    
    if new_col_name is not None:
        result.name = new_col_name
    return result
def vector_neut(target_col, ref_cols, new_col_name=None):
    """
    Neutralize target column with respect to reference columns using vector projection
    
    Parameters:
    target_col (pd.Series): Target column to neutralize
    ref_cols (list of pd.Series): List of reference columns
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with neutralized values
    """
    # Extract target vector and handle scalar values
    if isinstance(target_col, (int, float)):
        return target_col  # Can't neutralize a scalar with respect to vectors
        
    target = target_col.values.copy()  # Make a copy to avoid modifying the original
    
    # Process each reference column individually
    for ref_col in ref_cols:
        # Skip if reference column is a scalar
        if isinstance(ref_col, (int, float)):
            continue
            
        ref_vector = ref_col.values
        
        # Check for division by zero
        denominator = np.sum(ref_vector**2)
        if denominator > 0:
            # Calculate projection and subtract
            projection = np.dot(target, ref_vector) / denominator * ref_vector
            target = target - projection
    
    # Create result Series
    result = pd.Series(target, index=target_col.index)
    if new_col_name is not None:
        result.name = new_col_name
    return result

def regression_neut(target_col, ref_cols, new_col_name=None):
    """
    Neutralize target column with respect to reference columns using linear regression
    
    Parameters:
    target_col (pd.Series): Target column to neutralize
    ref_cols (list of pd.Series): List of reference columns
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with neutralized values
    """
    # Handle scalar target
    if isinstance(target_col, (int, float)):
        return target_col  # Can't neutralize a scalar with respect to vectors
    
    # Filter out scalar reference columns
    valid_ref_cols = []
    for ref_col in ref_cols:
        if not isinstance(ref_col, (int, float)):
            valid_ref_cols.append(ref_col)
    
    # If no valid reference columns, return target as is
    if not valid_ref_cols:
        result = target_col.copy()
        if new_col_name is not None:
            result.name = new_col_name
        return result
    
    # Create DataFrame with reference columns
    try:
        ref_df = pd.concat(valid_ref_cols, axis=1)
    except ValueError:
        # Handle case where concat fails (e.g., all Series are empty)
        result = target_col.copy()
        if new_col_name is not None:
            result.name = new_col_name
        return result
    
    # Combine ref_df with target_col for easier handling of missing values
    combined = pd.concat([ref_df, target_col], axis=1)
    
    # Remove rows with NaN values
    valid_data = combined.dropna()
    
    # If no valid rows, return Series with NaNs
    if len(valid_data) == 0:
        result = pd.Series(np.nan, index=target_col.index)
        if new_col_name is not None:
            result.name = new_col_name
        return result
    
    # Extract features and target
    X = valid_data.iloc[:, :-1]
    y = valid_data.iloc[:, -1]
    
    # Fit linear regression
    model = LinearRegression()
    
    # Handle case where X has only one column (reshape required)
    if X.shape[1] == 1:
        model.fit(X.values.reshape(-1, 1), y)
        
        # Make predictions
        valid_indices = valid_data.index
        X_for_predict = ref_df.loc[valid_indices].values.reshape(-1, 1)
        y_pred = model.predict(X_for_predict)
    else:
        model.fit(X, y)
        
        # Make predictions
        valid_indices = valid_data.index
        X_for_predict = ref_df.loc[valid_indices]
        y_pred = model.predict(X_for_predict)
    
    # Compute residuals
    residuals = target_col.loc[valid_indices] - y_pred
    
    # Initialize result with NaN
    result = pd.Series(np.nan, index=target_col.index)
    
    # Update only valid rows
    result.loc[valid_indices] = residuals
    
    if new_col_name is not None:
        result.name = new_col_name
    return result

def groupby(data, group_col, agg_col, agg_func='mean', new_col_name=None):
    """
    Aggregate a column by groups
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the columns
    group_col (pd.Series or list of pd.Series): Column(s) to group by
    agg_col (pd.Series): Column to aggregate
    agg_func (str or function, optional): Aggregation function ('mean', 'median', 'sum', 'std', etc.)
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with aggregated values
    """
    # Create a temporary DataFrame
    if isinstance(group_col, list):
        temp_df = pd.concat(group_col + [agg_col], axis=1)
    else:
        temp_df = pd.concat([group_col, agg_col], axis=1)
    
    # Get column names
    if isinstance(group_col, list):
        group_names = [col.name for col in group_col]
    else:
        group_names = [group_col.name]
    
    # Compute aggregation
    result = temp_df.groupby(group_names)[agg_col.name].transform(agg_func)
    
    if new_col_name is not None:
        result.name = new_col_name
    
    return result

def winsorize(col, std=4, new_col_name=None):
    """
    Winsorize a column to limit extreme values based on standard deviations
    
    Parameters:
    col (pd.Series): Column to transform
    std (float, optional): Number of standard deviations to use for clipping values
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with winsorized values
    """
    mean = col.mean()
    std_dev = col.std()
    lower_limit = mean - std * std_dev
    upper_limit = mean + std * std_dev
    
    result = col.clip(lower=lower_limit, upper=upper_limit)
    
    if new_col_name is not None:
        result.name = new_col_name
    return result

def bucket(col, buckets=None, range=None, new_col_name=None):
    """
    Convert values into bucket indexes
    
    Parameters:
    col (pd.Series): Column to transform
    buckets (str or list, optional): Comma-separated string or list of bucket boundaries
    range (str, optional): Range specified as "start, end, step"
    new_col_name (str, optional): Name for the new column
    
    Returns:
    pd.Series: Series with bucket indexes
    """
    if buckets is not None:
        # Handle string input
        if isinstance(buckets, str):
            buckets = [float(x.strip()) for x in buckets.split(',')]
        
        # Convert to list if it's another iterable
        buckets = list(buckets)
        
        # Ensure buckets are sorted
        buckets.sort()
        
        # Create bucket labels (one fewer than boundaries)
        labels = list(range(len(buckets) - 1))
        
        # Use pandas cut to create buckets
        result = pd.cut(col, bins=buckets, labels=labels, include_lowest=True)
        
        # Convert to integer Series
        result = pd.Series(result.astype('Int64'), index=col.index)
        
    elif range is not None:
        # Parse range string
        range_parts = [float(x.strip()) for x in range.split(',')]
        
        if len(range_parts) != 3:
            raise ValueError("Range should be specified as 'start, end, step'")
            
        start, end, step = range_parts
        
        # Create bucket boundaries
        bucket_boundaries = np.arange(start, end + step/2, step)
        
        # Create bucket labels
        labels = list(range(len(bucket_boundaries) - 1))
        
        # Use pandas cut to create buckets
        result = pd.cut(col, bins=bucket_boundaries, labels=labels, include_lowest=True)
        
        # Convert to integer Series
        result = pd.Series(result.astype('Int64'), index=col.index)
        
    else:
        raise ValueError("Either 'buckets' or 'range' must be specified")
    
    if new_col_name is not None:
        result.name = new_col_name
    return result