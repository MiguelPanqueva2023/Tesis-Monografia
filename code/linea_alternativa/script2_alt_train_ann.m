% =========================================================================
% PROYECTO: CONTROL HÍBRIDO NMPC + RED NEURONAL RESIDUAL (GREY-BOX)
% SCRIPT 2 (ALT): ENTRENAMIENTO MATRICIAL DE LA ANN RESIDUAL MULTICAPA
% LINEA: TRAYECTORIA VARIABLE INTERNACIONAL (COL -> ARG -> USA)
% OBJETIVO: Entrenar mediante Backpropagation manual una arquitectura 
%           MISO (5 entradas, 2 capas ocultas, 2 salidas) para identificar
%           los residuos dinámicos generados bajo la referencia dinámica.
% =========================================================================

clc; clear; close all;

%% 1. PREPARACIÓN DE DATOS E INTEGRIDAD DEL ENTORNO
archivo_data = 'dataset_hibrido_refdin.mat';
if ~exist(archivo_data, 'file')
    error('Falta el archivo "%s". Ejecute el Script 1 Alternativo primero.', archivo_data);
end
load(archivo_data);

% Filtrado analítico de ruido en la variable de control (Riego infinitesimal)
Xdata(abs(Xdata(:,3)) < 1e-4, 3) = 0;
n_total = size(Xdata, 1);

% Fijación de semilla estocástica para reproducibilidad académica
rng(42); 

% --- OPERACIÓN DE BARAJADO (SHUFFLE) ---
% Mezcla estocástica de vectores para desacoplar la correlación temporal lineal 
% y asegurar una correcta generalización en regímenes de crecimiento y senescencia.
idx_rand = randperm(n_total);
X_sh = Xdata(idx_rand, :);
Y_sh = Ydata(idx_rand, :);

p_train = 0.80; % Partición matricial: 80% Entrenamiento / 20% Validación
n_train = floor(p_train * n_total);

% Transposición de registros para compatibilidad con álgebra de operadores de la ANN
X_train = X_sh(1:n_train, :)';    Y_train = Y_sh(1:n_train, :)';    
X_test  = X_sh(n_train+1:end, :)'; Y_test  = Y_sh(n_train+1:end, :)'; 

%% 2. ESTANDARIZACIÓN POR ESCALAMIENTO MIN-MAX [0, 1]
xmin = min(X_train, [], 2); xmax = max(X_train, [], 2);
ymin = min(Y_train, [], 2); ymax = max(Y_train, [], 2);

% Inyección de eps (épsilon de máquina) para blindar el algoritmo ante divisiones por cero
Xn_train = (X_train - xmin) ./ (xmax - xmin + eps);
Yn_train = (Y_train - ymin) ./ (ymax - ymin + eps);
Xn_test  = (X_test - xmin) ./ (xmax - xmin + eps);

%% 3. ARQUITECTURA DE LA RED NEURONAL (5 -> 20 -> 10 -> 2)
n_in = 5; n_h1 = 20; n_h2 = 10; n_out = 2;

% Inicialización Normalizada de Xavier/Glorot para mitigar desvanecimiento de gradiente
W1 = randn(n_h1, n_in) * sqrt(2/n_in);   b1 = zeros(n_h1, 1);
W2 = randn(n_h2, n_h1) * sqrt(2/n_h1);  b2 = zeros(n_h2, 1);
W3 = randn(n_out, n_h2) * sqrt(2/n_h2); b3 = zeros(n_out, 1);

%% 4. BUCLE DE OPTIMIZACIÓN DETERMINÍSTICA (BACKPROPAGATION MANUAL)
epochs = 20000; 
lr = 0.01;      
mse_hist = zeros(epochs, 1);

fprintf('Iniciando el entrenamiento de la ANN Residual (%d Épocas)...\n', epochs);

for e = 1:epochs
    % --- OPERACIÓN FORWARD (PROPAGACIÓN HACIA ADELANTE) ---
    A1 = tanh(W1 * Xn_train + b1);   % Activación no lineal capa oculta 1
    A2 = tanh(W2 * A1 + b2);         % Activación no lineal capa oculta 2
    Yhat = W3 * A2 + b3;             % Salida de regresión lineal pura
    
    Error = Yn_train - Yhat;         
    mse_hist(e) = mean(Error(:).^2); % Métrica del Error Cuadrático Medio
    
    % --- OPERACIÓN BACKWARD (RETROPROPAGACIÓN DEL GRADIENTE) ---
    dY = -2 * Error / n_train;
    dW3 = dY * A2';        db3 = sum(dY, 2);
    
    dZ2 = (W3' * dY) .* (1 - A2.^2); % Derivada implícita de la función tangente hiperbólica
    dW2 = dZ2 * A1';       db2 = sum(dZ2, 2);
    
    dZ1 = (W2' * dZ2) .* (1 - A1.^2);
    dW1 = dZ1 * Xn_train';  db1 = sum(dZ1, 2);
    
    % --- ACTUALIZACIÓN PARAMÉTRICA POR GRADIENTE DESCENDENTE ---
    W3 = W3 - lr*dW3; b3 = b3 - lr*db3;
    W2 = W2 - lr*dW2; b2 = b2 - lr*db2;
    W1 = W1 - lr*dW1; b1 = b1 - lr*db1;
    
    if mod(e, 2500) == 0
        fprintf('  Época [%d/%d] -> MSE de Entrenamiento = %.8f\n', e, epochs, mse_hist(e));
    end
end

%% 5. EVALUACIÓN Y RECONSTRUCCIÓN DEL MODELO HÍBRIDO
% Mapeo temporal completo (Datos ordenados cronológicamente)
X_all_n = (Xdata' - xmin) ./ (xmax - xmin + eps);
Yhat_res_n = W3 * tanh(W2 * tanh(W1 * X_all_n + b1) + b2) + b3;
Yhat_res = Yhat_res_n .* (ymax - ymin + eps) + ymin; % Transformación inversa al espacio real

%% 6. BLOQUE GRÁFICO DE ALTA RIGUROSIDAD EDITORIAL
fs = 11; fs_tit = 12; black = [0.15 0.15 0.15];

% --- FIGURA 1: DINÁMICA DE CONVERGENCIA ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.5 0.4 0.4], 'Name', 'Curva de Aprendizaje');
semilogy(mse_hist, 'Color', [0.85 0.32 0.09], 'LineWidth', 2.5);
grid on;
xlabel('Épocas de Optimización', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black); 
ylabel('MSE (Escala Logarítmica)', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Dinámica de Convergencia del Backpropagation Manual', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);

% --- FIGURA 2: VALIDACIÓN TEMPORAL DE RESIDUOS FISIOLÓGICOS ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.5 0.1 0.45 0.7], 'Name', 'Validación de Residuos');
subplot(2,1,1); 
plot(Ydata(:,1), 'Color', [0 0.44 0.74], 'LineWidth', 2); hold on; 
plot(Yhat_res(1,:), 'r--', 'LineWidth', 1.5);
ylabel('Residuo de Humedad \Delta\theta', 'FontSize', fs, 'FontWeight', 'bold');
title('Identificación del Residuo Físico del Suelo', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on; set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs);
lgd_r1 = legend('Residuo Real (Planta-Nominal)', 'Predicción Red Neuronal', 'Location', 'northeast');
set(lgd_r1, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

subplot(2,1,2); 
plot(Ydata(:,2), 'Color', [0.46 0.67 0.18], 'LineWidth', 2); hold on; 
plot(Yhat_res(2,:), 'k--', 'LineWidth', 1.5);
ylabel('Residuo de Biomasa \DeltaB [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold');
xlabel('Tiempo (Días de Simulación)', 'FontSize', fs, 'FontWeight', 'bold');
title('Identificación del Residuo de Desarrollo Vegetativo', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on; set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs);
lgd_r2 = legend('Residuo Real (Planta-Nominal)', 'Predicción Red Neuronal', 'Location', 'northeast');
set(lgd_r2, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

% --- FIGURA 3: COMPARATIVA DE AJUSTE INTEGRAL (PREDICCIÓN GREY-BOX) ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.4 0.4], 'Name', 'Ajuste Grey-Box');
Biomasa_Real = Xdata(:,2); 
% Reconstrucción matemática: Descontamos la discrepancia no modelada mediante la predicción inteligente
Biomasa_Hibrida = Biomasa_Real - (Ydata(:,2) - Yhat_res(2,:)'); 

plot(Biomasa_Real, 'Color', [0 0 0.7], 'LineWidth', 2.5); hold on;
plot(Biomasa_Hibrida, 'r-.', 'LineWidth', 2);
ylabel('Biomasa Total Acumulada [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold');
xlabel('Tiempo [Días]', 'FontSize', fs, 'FontWeight', 'bold');
title('Validación de la Reconstrucción Híbrida (Caja Gris)', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on; set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);
lgd3 = legend('Dinámica de la Planta Real', 'Respuesta Híbrida Grey-Box (Nominal + ANN)', 'Location', 'northwest');
set(lgd3, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

% Guardado explícito de la sinapsis neuronal óptima
save('red_entrenada_refdin.mat', 'W1','b1','W2','b2','W3','b3','xmin','xmax','ymin','ymax');
fprintf('\nEntrenamiento finalizado. Estructura sináptica guardada en: "red_entrenada_refdin.mat"\n');