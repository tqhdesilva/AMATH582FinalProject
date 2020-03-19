from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler 
from sklearn.decomposition import PCA


pre_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=117),
)