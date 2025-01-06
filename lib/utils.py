import pickle

import numpy as np
import torch


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()



from sklearn.preprocessing import StandardScaler

""" Scale the log returns using StandardScaler only """
def scaling(data):
    """
    각 피처별로 StandardScaler를 적용하여 데이터를 스케일링합니다.

    Parameters:
        data (np.ndarray): 스케일링할 데이터, shape=(samples, features)

    Returns:
        scaled_data (np.ndarray): 스케일링된 데이터, shape=(samples, features)
        scalers (list of StandardScaler): 각 피처별로 학습된 StandardScaler 객체 리스트
    """
    scalers = [] 
    scaled_data = []

    for i in range(data.shape[1]):  # 피처별로 반복
        scaler = StandardScaler()

        # 현재 피처의 데이터 추출 및 스케일링
        feature_data = data[:, i].reshape(-1, 1)  # shape=(samples, 1)
        feature_scaled = scaler.fit_transform(feature_data)

        # 스케일러 및 스케일링된 데이터 저장
        scalers.append(scaler)
        scaled_data.append(feature_scaled.flatten())  # shape=(samples,)

    # 스케일링된 피처들을 다시 합쳐서 (samples, features) 형태로 변환
    scaled_data = np.array(scaled_data).T  # shape=(samples, features)
    return scaled_data, scalers


""" Inverse the scaling process for all features using StandardScaler only """
def inverse_scaling(y, scalers):
    """
    스케일링된 데이터를 원래대로 복원합니다.

    Parameters:
        y (torch.Tensor): 스케일링된 데이터, shape=(batch_size, features, seq_len)
        scalers (list of StandardScaler): 각 피처별로 학습된 StandardScaler 객체 리스트

    Returns:
        y_original (np.ndarray): 복원된 원본 데이터, shape=(batch_size, features, seq_len)
    """
    y = y.cpu().detach().numpy()  # torch.Tensor를 NumPy 배열로 변환
    y_original = np.zeros_like(y)  # 원본 데이터를 저장할 배열 초기화

    for idx in range(y.shape[1]):
        scaler = scalers[idx]
        
        y_feature = y[:, idx, :]  # shape=(batch_size, seq_len)

        # 각 피처별로 스케일링을 원상복구
        # reshape하여 StandardScaler가 요구하는 2D 형태로 변환
        y_feature_reshaped = y_feature.reshape(-1, 1)  # shape=(batch_size * seq_len, 1)
        y_feature_original = scaler.inverse_transform(y_feature_reshaped)
        
        # 다시 원래의 shape으로 복원
        y_original[:, idx, :] = y_feature_original.reshape(y_feature.shape)

    return y_original

def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)