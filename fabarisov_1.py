def set_eye_column(matrix, column, fixed_row):
    row_number = matrix.shape[0]
    for row_idx in range(row_number):
        if (row_idx != fixed_row and matrix[row_idx, column] != 0):
            matrix[row_idx] = (matrix[row_idx] + matrix[fixed_row]) % 2

def find_nonzero_row(matrix, column, next_row, except_rows):
    row_number = matrix.shape[0]
    while (matrix[next_row, column] == 0 or next_row in except_rows):
        next_row += 1
        if (next_row >= row_number):
            return row_number
    return next_row

def make_generator_matrix(H):
    row_number, column_number = H.shape
    cur_H = H.copy()
    H_eye_row_permutation = list()
    H_eye_columns = list()
    for column in range(column_number):
        nonzero_row = find_nonzero_row(cur_H, column, 0, H_eye_row_permutation)
        if (nonzero_row >= row_number):
            continue
        H_eye_row_permutation.append(nonzero_row) # сохраняем индексы строк
        H_eye_columns.append(column)  # сохраняем индексы столбцов
        set_eye_column(cur_H, column, nonzero_row)
    G_P_row_indexes = np.zeros(row_number)
    G_P_row_indexes[H_eye_row_permutation] = H_eye_columns
    G_eye_row_indexes = [i for i in range(column_number) if i not in G_P_row_indexes]
    P_width = column_number - row_number
    G = np.zeros((column_number, P_width))
    G[G_eye_row_indexes, :] = np.eye(P_width)
    G[G_P_row_indexes.astype(int), :] = cur_H[:, G_eye_row_indexes]
    return G, np.array(G_eye_row_indexes).astype(int)




def initialize(H, q):
    row_number, column_number = H.shape
    
    # из вершин в факторы
    mu_v_f = np.zeros((column_number, row_number, 2)) 
    mu_v_f[:, :, 1] = q * H.T
    mu_v_f[:, :, 0] = (1 - q) * H.T
    
    # из факторов в вершины
    mu_f_v = np.zeros((row_number, column_number, 2)) 
    
    beliefs = np.zeros((column_number, 2))
    old_beliefs = np.zeros((column_number, 2))
    
    e = np.zeros(column_number) # вектор текущих ошибок
    e_progress = list() # список векторов ошибок в зависимости от итерации
    stable_beliefs = list() # доля стабилизировавшихся ошибок в зависимости от итерации
    
    return row_number, column_number, mu_v_f, mu_f_v, beliefs, old_beliefs, e, e_progress, stable_beliefs

def update_mu_v_f(H, q, mu_f_v, mu_v_f, damping, i, j):
    # факторы-соседи i кроме j
    i_neighbors = [item for item in np.nonzero(H[:, i])[0] if item != j]
    
    mu_new = np.zeros(2)
    mu_new[0] = np.prod(mu_f_v[i_neighbors, i, 0]) * (1 - q)
    mu_new[1] = np.prod(mu_f_v[i_neighbors, i, 1]) * q
    mu_new /= np.sum(mu_new)
    
    mu_v_f[i, j, 0] = damping * mu_new[0] + (1 - damping) * mu_v_f[i, j, 0]
    mu_v_f[i, j, 1] = damping * mu_new[1] + (1 - damping) * mu_v_f[i, j, 1]

def update_mu_f_v(s, H, q, mu_f_v, mu_v_f, damping, j, i):
        
    # вершины, кроме i, входящие в фактор j
    j_neighbors = [item for item in np.nonzero(H[j])[0] if item != i] 
    
    delta_mu_v_f_j = mu_v_f[:, j, 0] - mu_v_f[:, j, 1]
    delta_p = np.prod([delta_mu_v_f_j[k] for k in j_neighbors])
    
    # выразили из системы
    p = 0.5 * np.array([1 + delta_p, 1 - delta_p]) 
    
    # пересчет сообщений (меньше damping - меньше обновляется)
    mu_f_v[j, i, 0] = damping * p[s[j]] + (1 - damping) * mu_f_v[j, i, 0]
    mu_f_v[j, i, 1] = damping * p[1 - s[j]] + (1 - damping) * mu_f_v[j, i, 1]

def update_beliefs(H, q, mu_f_v, beliefs, i):
    i_neighbors = np.nonzero(H[:, i])[0]
    beliefs[i, 0] = np.prod(mu_f_v[i_neighbors, i, 0]) * (1 - q)
    beliefs[i, 1] = np.prod(mu_f_v[i_neighbors, i, 1]) * q

def get_stable_beliefs_rate(beliefs, old_beliefs, tol_beliefs):
    diff = np.absolute(beliefs - old_beliefs)
    number_stable = len(np.where(diff < tol_beliefs)[0])
    return number_stable / len(old_beliefs) / 2.0

def update_e(H, s, old_beliefs, beliefs, display, tol_beliefs):
    e = np.argmax(beliefs, axis=1)
    result = 2
    if display:
        print 'e', e
    if np.all(H.dot(e) == s):
        return e, 0
    if np.linalg.norm(beliefs - old_beliefs) / np.linalg.norm(beliefs) < tol_beliefs:
        return e, 1
    return e, 2

def decode(s, H, q, schedule = 'parallel', damping = 1, max_iter = 40, tol_beliefs = 1e-4, display = False):
    
    row_number, column_number, mu_v_f, mu_f_v, beliefs, old_beliefs, e, e_progress, stable_beliefs = initialize(H, q)
    result = 2
        
    for iter in range(max_iter):
        if schedule == 'parallel':
            
            # все факторы посылают сообщения во все вершины
            flag = 1
            for j in range(row_number):
                for i in np.nonzero(H[j])[0]:
                    update_mu_f_v(s, H, q, mu_f_v, mu_v_f, damping, j, i)
                    
            old_beliefs = beliefs.copy()
            # все вершины посылают сообщения во все факторы
            for i in range(column_number):
                update_beliefs(H, q, mu_f_v, beliefs, i)
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_v_f(H, q, mu_f_v, mu_v_f, damping, i, j)

        if schedule == 'sequential':
            for i in range(column_number):
                # данной вершины сначала вычисляются все входящие сообщения от соседних факторов
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_f_v(s, H, q, mu_f_v, mu_v_f, damping, j, i)
                old_beliefs = beliefs.copy()
                update_beliefs(H, q, mu_f_v, beliefs, i)
                
                # вычисляются все исходящие сообщения в соседние факторы
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_v_f(H, q, mu_f_v, mu_v_f, damping, i, j)
        
        stable_beliefs.append(get_stable_beliefs_rate(beliefs, old_beliefs, tol_beliefs))
        e, result = update_e(H, s, old_beliefs, beliefs, display, tol_beliefs)
        e_progress.append(e)
        if result != 2:
            break
            
    return e, result

def estimate_errors(H, q, num_points = 40):
    success = 0.0
    err_bit = 0.0
    err_block = 0.0
    
    row_number, column_number = H.shape
    
    for i in range(num_points):
        e = get_noise(column_number, q)
        s = H.dot(e) % 2
        e_decoding, status, _, _ = decode(s, H, q)
        if status < 2:
            err_block += 1 - np.all(e_decoding == e)
            err_bit += np.sum(e_decoding != e) * 1.0 / column_number
            success += 1
        
    diver = (num_points - success) / num_points
    err_bit /= success
    err_block /= success
    
    return err_bit, err_block, diver