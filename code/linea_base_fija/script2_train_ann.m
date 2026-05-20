% =========================================================================
% SCRIPT 2: ENTRENAMIENTO HÍBRIDO CON ANÁLISIS DE CONVERGENCIA
% =========================================================================
clc; clear; close all;

%% 1. PREPARACIÓN DE DATOS
if ~exist('dataset_hibrido.mat', 'file'), error('Ejecuta el Script 1 primero.'); end
load dataset_hibrido.mat %Para ref estatica

% Limpiamos valores de riego insignificantes para evitar ruido en la ANN
Xdata(abs(Xdata(:,3)) < 1e-4, 3) = 0;

n_total = size(Xdata, 1);
rng(42); % Fijamos semilla para que siempre te den los mismos resultados

% --- SHUFFLE (BARAJADO) ---
% Mezclamos los datos para que la red vea días de crecimiento y días de 
% muerte (senescencia) mezclados en el entrenamiento.
idx_rand = randperm(n_total);
X_sh = Xdata(idx_rand, :);
Y_sh = Ydata(idx_rand, :);

p_train = 0.80; % 80% para entrenar, 20% para validar
n_train = floor(p_train * n_total);

% Separación en matrices de entrenamiento y prueba (transpuestas para la ANN)
X_train = X_sh(1:n_train, :)';    Y_train = Y_sh(1:n_train, :)';    
X_test  = X_sh(n_train+1:end, :)'; Y_test  = Y_sh(n_train+1:end, :)'; 

%% 2. NORMALIZACIÓN MIN-MAX [0, 1]
% La red converge mucho más rápido si los datos están en el mismo rango
xmin = min(X_train, [], 2); xmax = max(X_train, [], 2);
ymin = min(Y_train, [], 2); ymax = max(Y_train, [], 2);

Xn_train = (X_train - xmin) ./ (xmax - xmin + eps);
Yn_train = (Y_train - ymin) ./ (ymax - ymin + eps);
Xn_test  = (X_test - xmin) ./ (xmax - xmin + eps);

%% 3. DEFINICIÓN DE LA ARQUITECTURA (Input: 5 -> Hidden: 20 -> Hidden: 10 -> Out: 2)
n_in = 5; n_h1 = 20; n_h2 = 10; n_out = 2;
% Inicialización de Xavier/Glorot para evitar que los gradientes mueran
W1 = randn(n_h1, n_in)*sqrt(2/n_in);  b1 = zeros(n_h1, 1);
W2 = randn(n_h2, n_h1)*sqrt(2/n_h1); b2 = zeros(n_h2, 1);
W3 = randn(n_out, n_h2)*sqrt(2/n_h2); b3 = zeros(n_out, 1);

%% 4. BUCLE DE ENTRENAMIENTO (BACKPROPAGATION MANUAL)
epochs = 20000; 
lr = 0.01;      
mse_hist = zeros(epochs, 1); % AQUÍ SE GUARDA LA CONVERGENCIA

for e = 1:epochs
    % --- Paso hacia adelante (Forward) ---
    A1 = tanh(W1 * Xn_train + b1); % Capa oculta 1 con activación tangente hiperbólica
    A2 = tanh(W2 * A1 + b2);       % Capa oculta 2
    Yhat = W3 * A2 + b3;           % Capa de salida (lineal para regresión)
    
    Error = Yn_train - Yhat;       % Cálculo del error residual
    mse_hist(e) = mean(Error(:).^2); % Guardamos el error promedio de esta época
    
    % --- Paso hacia atrás (Backprop) ---
    dY = -2 * Error / n_train;
    dW3 = dY * A2';        db3 = sum(dY, 2);
    dZ2 = (W3' * dY) .* (1 - A2.^2); % Derivada de tanh
    dW2 = dZ2 * A1';       db2 = sum(dZ2, 2);
    dZ1 = (W2' * dZ2) .* (1 - A1.^2);
    dW1 = dZ1 * Xn_train';  db1 = sum(dZ1, 2);
    
    % --- Actualización de pesos (Gradiente Descendente) ---
    W3 = W3 - lr*dW3; b3 = b3 - lr*db3;
    W2 = W2 - lr*dW2; b2 = b2 - lr*db2;
    W1 = W1 - lr*dW1; b1 = b1 - lr*db1;
end

%% 5. RECONSTRUCCIÓN DE RESULTADOS
% Usamos los datos originales (ordenados por tiempo) para las gráficas
X_all_n = (Xdata' - xmin) ./ (xmax - xmin + eps);
Yhat_res_n = W3 * tanh(W2 * tanh(W1 * X_all_n + b1) + b2) + b3;
Yhat_res = Yhat_res_n .* (ymax - ymin + eps) + ymin; % Des-normalización

%% 6. BLOQUE DE GRÁFICAS
fs = 12;      % Tamaño de fuente base
fs_tit = 14;  % Tamaño de fuente títulos
black = [0.1 0.1 0.1];

% --- FIGURA 1: CONVERGENCIA ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.5 0.4], 'Name', 'Análisis de Entrenamiento');
semilogy(mse_hist, 'Color', [0.85 0.32 0.09], 'LineWidth', 2.5);
grid on; 
xlabel('Épocas', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black); 
ylabel('MSE (Escala Log)', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Curva de Convergencia de la Red Neuronal', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);

% --- FIGURA 2: RESIDUOS (Lo que la red aprendió a corregir) ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.6 0.7], 'Name', 'Validación de Residuos');

subplot(2,1,1); 
plot(Ydata(:,1), 'Color', [0 0.44 0.74], 'LineWidth', 2); hold on; 
plot(Yhat_res(1,:), 'r--', 'LineWidth', 1.5);
ylabel('Residuo \theta', 'FontSize', fs, 'FontWeight', 'bold');
title('Identificación del Residuo de Humedad', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs);
lgd_r1 = legend('Real', 'ANN', 'Location', 'northeast');
set(lgd_r1, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

subplot(2,1,2); 
plot(Ydata(:,2), 'Color', [0.46 0.67 0.18], 'LineWidth', 2); hold on; 
plot(Yhat_res(2,:), 'k--', 'LineWidth', 1.5);
ylabel('Residuo B [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold');
xlabel('Muestras (Días)', 'FontSize', fs, 'FontWeight', 'bold');
title('Identificación del Residuo de Biomasa', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs);
lgd_r2 = legend('Real', 'ANN', 'Location', 'northeast');
set(lgd_r2, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

% --- FIGURA 3: BIOMASA TOTAL (Modelo Híbrido Final) ---
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.6 0.5], 'Name', 'Modelo Híbrido');
Biomasa_Real = Xdata(:,2); 
Biomasa_Hibrida = Biomasa_Real - (Ydata(:,2) - Yhat_res(2,:)'); 

plot(Biomasa_Real, 'Color', [0 0 0.8], 'LineWidth', 2.5); hold on;
plot(Biomasa_Hibrida, 'r--', 'LineWidth', 2);
ylabel('Biomasa [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold');
xlabel('Tiempo [días]', 'FontSize', fs, 'FontWeight', 'bold');
title('Crecimiento Total de Biomasa (Híbrido)', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);
lgd3 = legend('Realidad (RK4)', 'Modelo Grey-Box', 'Location', 'northwest');
set(lgd3, 'Color', 'w', 'TextColor', black, 'EdgeColor', black);

% Ajuste para exportación
set(gcf, 'PaperPositionMode', 'auto');

save('red_entrenada_manual.mat', 'W1','b1','W2','b2','W3','b3','xmin','xmax','ymin','ymax');