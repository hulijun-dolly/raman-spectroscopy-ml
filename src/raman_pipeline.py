import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression


def generate_synthetic_raman_data(n_samples=300, n_wavenumbers=1200, n_classes=3, random_state=42):
    """Generate synthetic Raman spectra with class-specific peaks."""
    rng = np.random.RandomState(random_state)
    wavenumbers = np.linspace(500, 1800, n_wavenumbers)

    class_centers = {
        0: [700, 1000, 1300],
        1: [800, 1100, 1450],
        2: [600, 950, 1600],
    }

    labels = rng.randint(0, n_classes, size=n_samples)
    spectra = np.zeros((n_samples, n_wavenumbers))

    for i, label in enumerate(labels):
        base = 0.1 * rng.randn(n_wavenumbers)
        spectrum = base.copy()
        for peak in class_centers[label]:
            peak_width = 25 + rng.rand() * 20
            x = (wavenumbers - peak) / peak_width
            peak_component = np.exp(-0.5 * x**2)
            spectrum += (0.8 + rng.rand() * 1.5) * peak_component

        baseline = 0.05 * np.sin(np.linspace(0, 2 * np.pi, n_wavenumbers))
        spectrum += baseline
        spectrum += 0.02 * rng.randn(n_wavenumbers)
        spectra[i] = spectrum

    return spectra, labels, wavenumbers


def cosmic_ray_removal(spectra, threshold=7.0, window_size=5):
    """Simple spike detection/removal using local median deviation."""
    corrected = spectra.copy()
    for i in range(corrected.shape[0]):
        row = corrected[i]
        local_med = pd.Series(row).rolling(window=window_size, center=True, min_periods=1).median().to_numpy()
        local_std = pd.Series(row).rolling(window=window_size, center=True, min_periods=1).std().fillna(0).to_numpy()
        spikes = np.abs(row - local_med) > threshold * (local_std + 1e-8)
        if np.any(spikes):
            corrected[i, spikes] = local_med[spikes]
    return corrected


def asymmetric_least_squares(signal, lam=1e5, p=0.01, niter=10):
    """Baseline correction using asymmetric least squares."""
    L = len(signal)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).toarray()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = spsolve(W + lam * D.T.dot(D), w * signal)
        w = p * (signal > Z) + (1 - p) * (signal <= Z)
    return Z


def baseline_correction(spectra, lam=1e5, p=0.01, niter=10):
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        corrected[i] = spectra[i] - asymmetric_least_squares(spectra[i], lam=lam, p=p, niter=niter)
    return corrected


def smooth_spectra(spectra, window_length=11, polyorder=3):
    smoothed = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        if window_length % 2 == 0:
            window_length += 1
        smoothed[i] = savgol_filter(spectra[i], window_length=window_length, polyorder=polyorder)
    return smoothed


def normalize_spectra(spectra, method='area'):
    normalized = spectra.copy()
    if method == 'area':
        area = np.sum(normalized, axis=1, keepdims=True)
        area[area == 0] = 1
        normalized = normalized / area
    elif method == 'max':
        max_val = np.max(normalized, axis=1, keepdims=True)
        max_val[max_val == 0] = 1
        normalized = normalized / max_val
    elif method == 'l2':
        norm = np.linalg.norm(normalized, axis=1, keepdims=True)
        norm[norm == 0] = 1
        normalized = normalized / norm
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    return normalized


def preprocess_raman_spectra(spectra, cosmic_removal_flag=True, baseline_flag=True, smooth_flag=True, normalize_flag=True):
    X = spectra.copy()
    if cosmic_removal_flag:
        X = cosmic_ray_removal(X)
    if baseline_flag:
        X = baseline_correction(X)
    if smooth_flag:
        X = smooth_spectra(X)
    if normalize_flag:
        X = normalize_spectra(X)
    return X


def fit_pca_svm(X_train, y_train, X_test, n_components=30, C=10.0, gamma='scale'):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('svm', SVC(kernel='rbf', C=C, gamma=gamma, random_state=42))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def fit_plsda(X_train, y_train, X_test, n_components=15):
    labels = np.asarray(y_train)
    one_hot = np.zeros((len(labels), len(np.unique(labels))))
    for idx, label in enumerate(labels):
        one_hot[idx, label] = 1

    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X_train, one_hot)
    probs = pls.predict(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, pls


def fit_random_forest(X_train, y_train, X_test, n_estimators=300, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return preds, rf


def evaluate_predictions(y_true, y_pred, label_names=None):
    if label_names is None:
        label_names = [str(i) for i in sorted(np.unique(y_true))]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'confusion_matrix': cm,
        'classification_report': report,
    }


def run_experiment():
    X_raw, y, wavenumbers = generate_synthetic_raman_data()
    X_preprocessed = preprocess_raman_spectra(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    label_names = [f'Class {i}' for i in range(len(np.unique(y)))]

    svm_pred, svm_model = fit_pca_svm(X_train, y_train, X_test)
    pls_pred, pls_model = fit_plsda(X_train, y_train, X_test)
    rf_pred, rf_model = fit_random_forest(X_train, y_train, X_test)

    results = {
        'PCA+SVM': evaluate_predictions(y_test, svm_pred, label_names),
        'PLS-DA': evaluate_predictions(y_test, pls_pred, label_names),
        'Random Forest': evaluate_predictions(y_test, rf_pred, label_names),
    }

    for name, metrics in results.items():
        print(f'\n=== {name} ===')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'F1 Macro: {metrics["f1_macro"]:.4f}')
        print(metrics['classification_report'])
        print('Confusion Matrix:')
        print(metrics['confusion_matrix'])

    return {
        'wavenumbers': wavenumbers,
        'X_preprocessed': X_preprocessed,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'results': results,
        'svm_model': svm_model,
        'pls_model': pls_model,
        'rf_model': rf_model,
    }


if __name__ == '__main__':
    run_experiment()
