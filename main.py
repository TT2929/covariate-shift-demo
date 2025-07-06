import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import os
import logging

#logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def dataset(fixed_x1_shift,
        fixed_x2_shift,
        random_shift_magnitude,
        rotation_angle_deg):
    rng = np.random.default_rng(1)

    #data point
    num_points = 1000

    #y=x
    x1 = rng.uniform(-10, 10, num_points).reshape(-1, 1)
    x2 = x1 + rng.normal(0, 3, num_points).reshape(-1, 1)

    #label
    y = np.where(x2 > 0, 1, 0).squeeze()

    #noise
    for i in range(num_points // 10):
        index = rng.integers(0, num_points)
        if y[index] == 1:
            x2[index] = rng.uniform(-4, 0)
        else:
            x2[index] = rng.uniform(0, 4)

    x = np.hstack((x1, x2))

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
    
    #---1.rotation---
    angle_rad = math.radians(rotation_angle_deg)
    rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]])
    x_test = np.dot(x_test, rotation_matrix.T)

    #---2.shift---
    x_test[:, 0] += fixed_x1_shift
    x_test[:, 1] += fixed_x2_shift

    #---3.random shift---
    x1_rand_shift = rng.uniform(-random_shift_magnitude, random_shift_magnitude)
    x2_rand_shift = rng.uniform(-random_shift_magnitude, random_shift_magnitude)
    x_test[:, 0] += x1_rand_shift
    x_test[:, 1] += x2_rand_shift

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test= scaler.transform(x_test)


    logging.info(f"train_dataset: {len(x_train)}")
    logging.info(f"test_dataset: {len(x_test)}")
    
    return x_train, x_test, y_train, y_test

def prepare_train_svm(x, y, weight):
    if len(y.shape) != 1:
        raise RuntimeError(f'`y` must be a vector (given shape {y.shape})')
    
    if x is not None:
        if len(x.shape) != 2:
            raise RuntimeError(f'`x` must be a matrix (given shape {x.shape})')
    
        n, d = x.shape
        if y.size != n:
            raise RuntimeError(f'`y` must be a vector of size {n} (given shape {y.shape})')
    else:
        d = None
        n = y.size
    
    if weight.shape != (n,):
        raise RuntimeError(f'Size of `weight` must be the number of training samples (given shape {weight.shape})')

    return n, d, x, y, weight


def rbf_kernel(x1, x2):
    if len(x1.shape) != 2 or len(x2.shape) != 2:
        raise RuntimeError('Matrices required')     
       
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d1 = x1.shape[1]
    d2 = x2.shape[1]
    if d1 != d2:
        raise RuntimeError('Dimensions not matching')
    
    dist = np.empty((n1, n2))
    for i in range(n1):
        dist[i, :] = np.linalg.norm(x2 - x1[i, :], axis=1)
    gamma = 1 / (x_train.shape[1] * x_train.var())
    k = np.exp(-gamma * dist**2)
    return k

def weight_KDE(x_train, x_test, h):
    kde_train = KernelDensity(kernel='gaussian', bandwidth=h).fit(x_train)
    kde_test = KernelDensity(kernel='gaussian', bandwidth=h).fit(x_test)
    #dentisy of train on train data
    P_tr_on_tr = np.exp(kde_train.score_samples(x_train))
    #dentisy of test on train data
    P_te_on_tr = np.exp(kde_test.score_samples(x_train))
    epsilon = 1e-10
    w = P_te_on_tr / (P_tr_on_tr + epsilon)

    return w

def weight_KMM(x_train, x_test, gamma_val, B=100.0, reg_param=1e-3):
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    K_tt = rbf_kernel(x_train, x_train)
    K_te_tr = rbf_kernel(x_test, x_train)
    H = K_tt + reg_param * np.eye(n_train)
    g = -np.mean(K_te_tr, axis=0)

    try:
        w = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        logging.warning("KMM: Singular matrix in np.linalg.solve. Adding more regularization.")
        H = K_tt + (reg_param * 10) * np.eye(n_train)
        w = np.linalg.solve(H, -g)

    w[w < 0] = 0 
    w[w > B] = B

    return w

def predict(alpha, rho, k):
    pred = np.dot(k, alpha) + rho
    ypred = np.where(pred > 0, 1, -1)
    return pred, ypred

def make_meshgrid(x_test):
    x1_min, x1_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    x2_min, x2_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
    x = np.c_[x1.ravel(), x2.ravel()]
    return x, x1, x2



def train_svm(x_train, y_train, lam, weight, gamma_val):
    _, _, x, y, weight = prepare_train_svm(x_train, y_train, weight)

    C_val = 1.0 / lam
    model = SVC(kernel='rbf', C=C_val, gamma=gamma_val)
    model.fit(x, y, sample_weight=weight)
    
    return model


#---Streamlit---
st.set_page_config(layout="wide") 

st.title("共変量シフト下での重み付きSVMデモンストレーション")

if 'test_data_center_x1' not in st.session_state:
    st.session_state.test_data_center_x1 = 0.0
if 'test_data_center_x2' not in st.session_state:
    st.session_state.test_data_center_x2 = 0.0
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0.0
if 'weight_method' not in st.session_state:
    st.session_state.weight_method = 'KDE'

#---sidebar---
st.sidebar.header("テストデータ分布コントロール")

if st.sidebar.button("テストデータ位置をリセット"):
    st.session_state.test_data_center_x1 = 0.0
    st.session_state.test_data_center_x2 = 0.0
    st.session_state.rotation_angle = 0.0
    st.session_state.random_shift_magnitude_slider = 1.0
    st.session_state.weight_method = 'KDE'
    st.rerun() 

#---slider---
st.session_state.test_data_center_x1 = st.sidebar.slider("テスト分布中心 (X1)", -10.0, 10.0, st.session_state.test_data_center_x1, 0.1)
st.session_state.test_data_center_x2 = st.sidebar.slider("テスト分布中心 (X2)", -10.0, 10.0, st.session_state.test_data_center_x2, 0.1)
st.session_state.rotation_angle = st.sidebar.slider("テスト分布回転角度 (度)", -90.0, 90.0, st.session_state.rotation_angle, 5.0)

st.sidebar.subheader("重み計算方法") 
st.session_state.weight_method = st.sidebar.radio(
    "select of weight method:", 
    ('KDE', 'KMM'),
    index=['KDE', 'KMM'].index(st.session_state.weight_method) 
)

x_train, x_test, y_train, y_test = dataset(
    fixed_x1_shift=st.session_state.test_data_center_x1,
    fixed_x2_shift=st.session_state.test_data_center_x2,
    random_shift_magnitude=1.0, 
    rotation_angle_deg=st.session_state.rotation_angle )
    
#---weight---
lam = 1.05
C_val = 1.0 / lam
gamma_val = 1.0 / (x_train.shape[1] * np.var(x_train)) 
h_bandwidth = 0.5 

weights = None

if st.session_state.weight_method == 'KDE':
    weights = weight_KDE(x_train, x_test, h_bandwidth)
    weights_normalized = weights * (len(x_train) / np.sum(weights))
    weight_method_display = "KDE" 
elif st.session_state.weight_method == 'KMM':
    weights = weight_KMM(x_train, x_test, gamma_val=gamma_val, B=100.0, reg_param=1e-3)
    weights_normalized = weights * (len(x_train) / np.sum(weights))
    weight_method_display = "KMM" 
else:
    st.error("無効な重み計算方法が選択されました。")
    weights_normalized = np.ones(len(x_train)) 
    weight_method_display = "デフォルト (重みなし)"

#---model---
svm_model = train_svm(x_train, y_train, lam, weights_normalized, gamma_val) 

#---plot---
x_combined_for_mesh = np.vstack((x_train, x_test))
x1_min, x1_max = x_combined_for_mesh[:, 0].min() - 1, x_combined_for_mesh[:, 0].max() + 1
x2_min, x2_max = x_combined_for_mesh[:, 1].min() - 1, x_combined_for_mesh[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
xy_grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_raw_decision = svm_model.decision_function(xy_grid_points)
prob = (1 / (1 + np.exp(-Z_raw_decision))).reshape(xx.shape) 

fig, ax = plt.subplots(figsize=(3, 3))
ax.set_title(f"distribution (center of test data: X1={st.session_state.test_data_center_x1:.1f}, X2={st.session_state.test_data_center_x2:.1f})")
    
ax.contour(xx, yy, Z_raw_decision.reshape(xx.shape), levels=[0], colors='black', linestyles='--', linewidths=2)
#train data
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', s=5, marker='o', alpha=0.1, label='Train Data')
#test data
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='bwr', s=10, marker='x', alpha=0.6, label='Test Data')
    
ax.grid(True)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
st.pyplot(fig) 


st.sidebar.subheader("result")    
train_acc = accuracy_score(y_train, svm_model.predict(x_train))
test_acc = accuracy_score(y_test, svm_model.predict(x_test))
st.sidebar.write(f"trian acc: **{train_acc:.2f}**")
st.sidebar.write(f"test acc: **{test_acc:.2f}**")
