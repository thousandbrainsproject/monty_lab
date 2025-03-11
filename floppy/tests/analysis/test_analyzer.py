import ast
from pathlib import Path

import pytest

from floppy.analysis.analyzer import AnalysisResult, CodeAnalyzer


def test_analyzer_basic():
    code = """
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

def process_data(data):
    scaled = StandardScaler().fit_transform(data)
    mean = np.mean(scaled, axis=0)
    pvalue = stats.ttest_1samp(scaled, 0)[1]
    return mean, pvalue
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert "numpy" in result.imports
    assert "scipy.stats" in result.imports
    assert "sklearn.preprocessing" in result.imports

    assert any("StandardScaler" in call for call in result.sklearn_calls)
    assert any("mean" in call for call in result.numpy_calls)
    assert any("ttest_1samp" in call for call in result.scipy_calls)


def test_analyzer_multiple_files(tmp_path):
    # Create temporary Python files for testing
    file1 = tmp_path / "script1.py"
    file1.write_text("""
import numpy as np
x = np.array([1, 2, 3])
""")

    file2 = tmp_path / "script2.py"
    file2.write_text("""
from scipy import stats
p_value = stats.ttest_1samp([1, 2, 3], 0)[1]
""")

    analyzer = CodeAnalyzer()
    results = analyzer.analyze_files([file1, file2])

    assert len(results) == 2
    assert "numpy" in results[0].imports
    assert "scipy.stats" in results[1].imports


def test_analyzer_with_errors():
    code_with_syntax_error = """
import numpy as np
if True
    x = np.array([1, 2, 3])
"""
    analyzer = CodeAnalyzer()
    with pytest.raises(SyntaxError):
        analyzer.analyze_code(code_with_syntax_error)


def test_analyzer_complex_usage():
    code = """
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_model(X, y):
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check normality
    _, pvalue = stats.normaltest(X_scaled)
    
    # Create and train model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_scaled, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    mean_imp = np.mean(importance)
    
    return rf, mean_imp, pvalue

# Data manipulation
data = np.random.randn(100, 4)
labels = np.random.randint(0, 2, 100)

model, importance, p = train_model(data, labels)
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    # Check imports
    assert "numpy" in result.imports
    assert "scipy" in result.imports
    assert "sklearn.ensemble" in result.imports
    assert "sklearn.preprocessing" in result.imports

    # Check function calls
    assert any("StandardScaler" in call for call in result.sklearn_calls)
    assert any("RandomForestClassifier" in call for call in result.sklearn_calls)
    assert any("normaltest" in call for call in result.scipy_calls)
    assert any("random.randn" in call for call in result.numpy_calls)


def test_analyzer_nested_structures():
    code = """
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
    
    def process(self, data):
        self.data = np.array(data)
        scaled = self.scaler.fit_transform(self.data)
        
        results = []
        for col in range(scaled.shape[1]):
            col_data = scaled[:, col]
            mean = np.mean(col_data)
            _, pval = ttest_1samp(col_data, 0)
            results.append((mean, pval))
        
        return results

processor = DataProcessor()
results = processor.process([[1, 2], [3, 4]])
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    # Check that we catch all the library calls in nested structures
    assert any("StandardScaler" in call for call in result.sklearn_calls)
    assert any("array" in call for call in result.numpy_calls)
    assert any("mean" in call for call in result.numpy_calls)
    assert any("ttest_1samp" in call for call in result.scipy_calls)


def test_analyzer_with_comments_and_docstrings():
    code = '''
"""
This module uses various scientific computing libraries:
- numpy for numerical operations
- scipy for statistical tests
- sklearn for machine learning
"""
import numpy as np
from scipy import stats

def analyze_data(data):
    """
    Analyze data using numpy and scipy.
    
    Parameters:
        data: numpy array to analyze
    """
    # Convert to numpy array if needed
    data = np.array(data)
    
    # Compute basic statistics
    mean = np.mean(data)  # Using numpy's mean function
    
    # Perform statistical test
    _, pval = stats.normaltest(data)  # Test for normality
    
    return mean, pval
'''
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert "numpy" in result.imports
    assert "scipy.stats" in result.imports
    assert any("array" in call for call in result.numpy_calls)
    assert any("mean" in call for call in result.numpy_calls)
    assert any("normaltest" in call for call in result.scipy_calls)


def test_analyzer_invalid_file():
    analyzer = CodeAnalyzer()
    with pytest.raises(FileNotFoundError):
        analyzer.analyze_files(["nonexistent_file.py"])


def test_analyzer_empty_file(tmp_path):
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    analyzer = CodeAnalyzer()
    result = analyzer.analyze_files([empty_file])

    assert len(result) == 1
    assert not result[0].imports
    assert not result[0].numpy_calls
    assert not result[0].scipy_calls
    assert not result[0].sklearn_calls


def test_analyzer_aliased_nested_imports():
    code = """
import numpy.random as rnd
import numpy.linalg as la
from scipy.stats import norm as normal_dist
from sklearn.ensemble.forest import RandomForestRegressor as RFR

data = rnd.randn(100)
matrix = rnd.rand(10, 10)
eigenvals = la.eigvals(matrix)
prob = normal_dist.cdf(data)
model = RFR(n_estimators=100)
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert "numpy.random" in result.imports
    assert "numpy.linalg" in result.imports
    assert "scipy.stats" in result.imports
    assert "sklearn.ensemble.forest" in result.imports

    assert any("randn" in call for call in result.numpy_calls)
    assert any("rand" in call for call in result.numpy_calls)
    assert any("eigvals" in call for call in result.numpy_calls)
    assert any("cdf" in call for call in result.scipy_calls)
    assert any("RandomForestRegressor" in call for call in result.sklearn_calls)


def test_analyzer_method_chaining():
    code = """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Complex method chaining
X = (np.array([[1, 2], [3, 4]])
     .reshape(-1, 2)
     .transpose()
     .dot(np.ones(2)))

# Sklearn pipeline with method chaining
pipe = (Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])
        .fit(X)
        .transform(X))
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert any("array" in call for call in result.numpy_calls)
    assert any("reshape" in call for call in result.numpy_calls)
    assert any("transpose" in call for call in result.numpy_calls)
    assert any("dot" in call for call in result.numpy_calls)
    assert any("Pipeline" in call for call in result.sklearn_calls)
    assert any("StandardScaler" in call for call in result.sklearn_calls)
    assert any("PCA" in call for call in result.sklearn_calls)


def test_analyzer_with_type_annotations():
    code = """
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from scipy.sparse import spmatrix

def process_data(
    data: NDArray[np.float64],
    sparse_data: spmatrix,
    model: BaseEstimator
) -> tuple[NDArray[np.float64], float]:
    processed = np.mean(data, axis=0)
    sparse_sum = sparse_data.sum()
    prediction = model.predict(processed)
    return processed, float(prediction)
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert "numpy" in result.imports
    assert "numpy.typing" in result.imports
    assert "sklearn.base" in result.imports
    assert "scipy.sparse" in result.imports
    assert any("mean" in call for call in result.numpy_calls)


def test_analyzer_with_context_managers():
    code = """
import numpy as np
from numpy.random import default_rng

# Test numpy's random number generator context
with np.random.default_rng(42) as rng:
    data = rng.normal(0, 1, 1000)
    shuffled = rng.permutation(data)

# Test numpy's error handling context
with np.errstate(divide='ignore'):
    result = np.log(0)
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert any("default_rng" in call for call in result.numpy_calls)
    assert any("normal" in call for call in result.numpy_calls)
    assert any("permutation" in call for call in result.numpy_calls)
    assert any("errstate" in call for call in result.numpy_calls)
    assert any("log" in call for call in result.numpy_calls)


def test_analyzer_with_comprehensions_and_generators():
    code = """
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm

# List comprehension with multiple library calls
data = [np.array([i, i**2]) for i in range(10)]

# Generator expression with library calls
probabilities = (norm.cdf(x) for x in np.linspace(-3, 3, 100))

# Dictionary comprehension
scalers = {
    name: cls() 
    for name, cls in [('standard', StandardScaler), ('minmax', MinMaxScaler)]
}

# Nested comprehensions
matrices = [[np.zeros((2, 2)) for _ in range(3)] for _ in range(2)]
"""
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_code(code)

    assert any("array" in call for call in result.numpy_calls)
    assert any("linspace" in call for call in result.numpy_calls)
    assert any("zeros" in call for call in result.numpy_calls)
    assert any("cdf" in call for call in result.scipy_calls)
    assert any("StandardScaler" in call for call in result.sklearn_calls)
    assert any("MinMaxScaler" in call for call in result.sklearn_calls)
