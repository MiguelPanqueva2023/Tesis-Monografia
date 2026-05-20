% =========================================================================
% PROYECTO: CONTROL HÍBRIDO NMPC + RED NEURONAL RESIDUAL (GREY-BOX)
% SCRIPT 3: VALIDACIÓN DEL CONTROLADOR GREY-BOX NMPC (LAZO CERRADO)
% LINEA: REFERENCIA ESTÁTICA
% OBJETIVO: Evaluar el desempeño del NMPC Grey-Box en lazo cerrado frente a
%           una estrategia de riego tradicional (On-Off por umbral) bajo
%           perturbaciones paramétricas e incertidumbre climática real.
% =========================================================================

clc; clear; close all;
import casadi.*

%% 0. CONFIGURACIÓN DE RUTA DE CASADI
casadi_path = 'C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b';
if ~exist(casadi_path, 'dir')
    error('No se encontró la ruta de CasADi. Ajusta casadi_path en el script.');
end
addpath(genpath(casadi_path));

%% 1. CARGA DE LA PARÁMETROS DE LA RED NEURONAL OPTIMIZADA
if ~exist('red_entrenada_manual.mat', 'file')
    error('Debe ejecutar primero el Script 2 para generar "red_entrenada_manual.mat".');
end
load('red_entrenada_manual.mat'); 

%% 2. LECTURA DE REGISTROS METEOROLÓGICOS IDEAM Y CÁLCULO DE ETo (FAO-56)
% Conservación de escenarios de validación (Descomentar según la simulación deseada)
archivo = 'Datos_IDEAM_Tibaitata.xlsx';
%archivo = 'Datos_IDEAM_Tibaitata_LLUVIA.xlsx'; 
%archivo = 'Datos_IDEAM_Tibaitata_2015_SEQUIA_AJUSTADA'; 

tabla = readtable(archivo);

% Extracción y conversión de registros meteorológicos de la tabla Excel
Tmax   = str2double(string(tabla.Tmax));
Tmin   = str2double(string(tabla.Tmin));
HR     = str2double(string(tabla.HR));
brillo = str2double(string(tabla.Brillo));
P_excel = str2double(string(tabla.P));

if isdatetime(tabla.Fecha)
    fechas = tabla.Fecha;
else
    fechas = datetime(tabla.Fecha);
end
nTot = height(tabla);
J_dias = day(fechas, 'dayofyear');

% Inicialización geográfica y pre-cálculo de la serie de Evapotranspiración (ETo)
altitud = 2550; latitud  = 4.69; u2 = 2.0; G = 0;
ETo_serie = zeros(nTot, 1);

for t = 1:nTot
    [Tm, gam, delt] = modulo_psicrometria(Tmax(t), Tmin(t), altitud);
    [es, ea]        = modulo_vapor(Tmax(t), Tmin(t), HR(t));
    Rn              = modulo_radiacion(brillo(t), J_dias(t), latitud, Tmax(t), Tmin(t), ea);
    
    num = 0.408 * delt * (Rn - G) + gam * (900 / (Tm + 273)) * u2 * (es - ea);
    den = delt + gam * (1 + 0.34 * u2);
    ETo_serie(t) = num / den;
    
    if ~isfinite(ETo_serie(t)) || ETo_serie(t) < 0
        ETo_serie(t) = 0;
    end
end

%% 3. PARÁMETROS ESTRUCTURALES DEL SISTEMA (ALINEADOS CON SCRIPTS 1 Y 2)
theta_fc = 0.36;      % Humedad a Capacidad de Campo
theta_pwp = 0.18;     % Humedad a Punto de Marchitez Permanente
p_agotamiento = 0.35; % Fracción de agotamiento permisible del agua útil
Zr = 0.40;            % Profundidad radicular efectiva [m]
theta_c = theta_fc - p_agotamiento * (theta_fc - theta_pwp);
Kd = 0.6;             % Coeficiente de drenaje aparente
alpha = 60;           % Parámetro de suavizado de la función softplus
Kc_ini = 0.45; Kc_mid = 1.15; Kc_fin = 0.75;
Bmax = 1500;          % Biomasa máxima potencial [g/m^2]
r_grow = 0.12;        % Tasa intrínseca de crecimiento logístico [día^-1]
theta_ref = 0.32;     % Humedad objetivo (Referencia de control)
Imax = 6;             % Límite superior de la lámina de riego [mm/día]

%% 4. MODELADO NOMINAL Y ACOPLAMIENTO DE LA ANN RESIDUAL EN CASADI
x = SX.sym('x', 2);  % [Humedad volumétrica; Biomasa]
u = SX.sym('u');     % Lámina de riego artificial
p = SX.sym('p', 2);  % Perturbaciones climáticas: [Lluvia; ETo]
t_sym = SX.sym('t'); % Tiempo de simulación continuo [Días]

theta = x(1); B = x(2);

% Lógicas nominales de desarrollo (Kc) y restricción hídrica (Ks)
Kc_nom = if_else(B < 0.9 * Bmax, ...
    Kc_ini + (Kc_mid - Kc_ini) * (B / Bmax), ...
    Kc_mid - (Kc_mid - Kc_fin) * (1 - B / Bmax));
Ks_nom = fmin(1, fmax(0, (theta - theta_pwp) / (theta_c - theta_pwp)));

% Ecuaciones constitutivas de balance de masa exógeno (Nominal)
D_vol_nom = (Kd / alpha) * log(1 + exp(alpha * (theta - theta_fc)));
ETc_nom = Ks_nom * Kc_nom * p(2);
delta_s_nom = 0.01 + 0.015 / (1 + exp(-0.15 * (t_sym - 105)));

% Integración de lluvia continua suavizada
P_suave = calcular_lluvia_suave(p(1), t_sym, 30, 0.4, 0.7);

dtheta_nom = (u + P_suave - ETc_nom) / (1000 * Zr) - D_vol_nom;
dB_nom = (r_grow * Ks_nom) * B * (1 - B / Bmax) - delta_s_nom * B;
rhs_phys = [dtheta_nom; dB_nom];
f_ode_nom = Function('f_ode_nom', {x, u, p, t_sym}, {rhs_phys});

% --- ACOPLAMIENTO DIRECTO DE LA RED NEURONAL PERCEPTRÓN MULTICAPA ---
input_nn = [x; u; p];
input_n = (input_nn - xmin) ./ (xmax - xmin + eps); % Normalización Min-Max in-line
a1 = tanh(W1 * input_n + b1);
a2 = tanh(W2 * a1 + b2);
res_n = W3 * a2 + b3;
res = res_n .* (ymax - ymin + eps) + ymin; % Des-normalización in-line del residuo estimado

% --- INTEGRACIÓN RUNGE-KUTTA 4 (DINÁMICA NOMINAL) ---
h = 1; 
k1 = f_ode_nom(x, u, p, t_sym);
k2 = f_ode_nom(x + h/2*k1, u, p, t_sym + h/2);
k3 = f_ode_nom(x + h/2*k2, u, p, t_sym + h/2);
k4 = f_ode_nom(x + h*k3, u, p, t_sym + h);
x_next_phys_rk4 = x + h/6*(k1 + 2*k2 + 2*k3 + k4);

% --- ESTRUCTURACIÓN DEL MODELO CAJA GRIS HÍBRIDO ---
x_next_grey = x_next_phys_rk4 + res; 
f_greybox = Function('f_greybox', {x, u, p, t_sym}, {x_next_grey});
f_nominal = Function('f_nominal', {x, u, p, t_sym}, {x_next_phys_rk4});

%% 5. MODELADO DE LA PLANTA REAL PERTURBADA (VALIDACIÓN CIEGA)
f_real_ode = @(tt, xx, uu, pp) planta_real_ode( ...
    tt, xx, uu, pp, ...
    theta_fc, theta_pwp, theta_c, Bmax, Zr, Kc_ini, Kc_mid, Kc_fin, Kd, alpha);

%% 6. CONFIGURACIÓN FORMAL DEL CONTROLADOR OPTIMIZADOR NMPC
N = 10;  % Horizonte de predicción extendido a 10 días
X = SX.sym('X', 2, N+1);
U = SX.sym('U', N);
X0 = SX.sym('X0', 2);
P_h = SX.sym('P_h', N);
ETo_h = SX.sym('ETo_h', N);
t0 = SX.sym('t0');

% Configuración del vector agregado de optimización del problema NLP
w = [reshape(X, 2*(N+1), 1); U];
J = 0;
g = [X(:,1) - X0];

for k = 1:N
    % Conservación de funciones de costo estructuradas para alternación de escenarios:
    J = J + 2500 * (X(1,k+1) - theta_ref)^2 + 0.05 * U(k)^2 - 0.01 * X(2,k+1); % Escenario Normal Original
    %J = J + 1200*(X(1,k+1) - 0.32)^2 + 20.0*U(k)^2 - 0.001*X(2,k+1); %% Escenario Lluvia/Sequía (6mm y 3.5mm)
    %J = J + 5000*(X(1,k+1) - 0.32)^2 + 0.001*U(k)^2 - 0.01*X(2,k+1); % Escenario Sequía Crítica (2mm)
    
    % Inyección de las restricciones de igualdad basadas en el predictor Caja Gris Híbrido
    g = [g; X(:,k+1) - f_greybox(X(:,k), U(k), [P_h(k); ETo_h(k)], t0 + k - 1)];
end

% Ajustes paramétricos del resolvedor NLP (IPOPT)
opts = struct;
opts.ipopt.print_level = 0;
opts.ipopt.max_iter = 200;
opts.ipopt.tol = 1e-6;
opts.ipopt.acceptable_tol = 1e-4;
opts.print_time = 0;
solver = nlpsol('solver', 'ipopt', struct('x', w, 'f', J, 'g', g, 'p', [X0; t0; P_h; ETo_h]), opts);

%% 7. LÍMITES OPERATIVOS Y RESTRICCIONES DE BORDE
lbX = zeros(2*(N+1), 1); ubX = zeros(2*(N+1), 1);
for k = 1:(N+1)
    idx = 2*(k-1) + 1;
    lbX(idx)   = theta_pwp;  
    ubX(idx)   = 0.45;       % Margen que permite incrementos por precipitación sin saturar el control
    lbX(idx+1) = 0;          
    ubX(idx+1) = Bmax;
end
lbU = zeros(N, 1);
ubU = Imax * ones(N, 1);

lbx = [lbX; lbU]; ubx = [ubX; ubU];
lbg = zeros(2*(N+1), 1); ubg = zeros(2*(N+1), 1);

%% 8. SIMULACIÓN EN LAZO CERRADO DE LA ARQUITECTURA HÍBRIDA
Tsim = min(150, nTot - N);

% Inicialización de matrices históricas de almacenamiento
theta_real = zeros(Tsim,1); B_real     = zeros(Tsim,1); u_hist     = zeros(Tsim,1);
theta_pred = zeros(Tsim,1); B_pred     = zeros(Tsim,1);
theta_nom  = zeros(Tsim,1); B_nom      = zeros(Tsim,1);

% Establecimiento de estados iniciales
x_real = [0.30; 5.0];
x_nom_state  = x_real; 
x_pred_model = x_real;

% Inicialización estructurada del vector de solución inicial (Warm Start)
w0 = zeros(length(lbx), 1);
for k = 1:(N+1)
    idx = 2*(k-1) + 1;
    w0(idx)   = x_real(1); w0(idx+1) = x_real(2);
end
w0(end-N+1:end) = 1.0;

fprintf('Iniciando simulación NMPC Grey-Box por %d días...\n', Tsim);
for t = 1:Tsim
    % Extracción secuencial del vector de perturbaciones futuras en el horizonte
    P_hor   = P_excel(t:t+N-1);
    ETo_hor = ETo_serie(t:t+N-1);
    P_hor(isnan(P_hor)) = 0; 
    ETo_hor(isnan(ETo_hor)) = 0;
    
    % Ejecución recursiva del controlador NMPC
    try
        sol = solver('x0', w0, 'lbx', lbx, 'ubx', ubx, 'lbg', lbg, 'ubg', ubg, 'p', [x_real; t; P_hor; ETo_hor]);
        w_opt = full(sol.x);
        u_k = w_opt(end-N+1);  % Extracción de la acción inmediata de control (Horizonte Recesivo)
        u_k = min(max(u_k, 0), Imax);
        w0 = w_opt;            % Retención de estados óptimos para el siguiente Warm Start
    catch ME
        warning('NMPC no convergió en el paso t=%d. Se aplica u=0 de seguridad. Detalles: %s', t, ME.message);
        u_k = 0;
    end
    
    % Captura de variables climatológicas instantáneas del día actual
    P_t = P_excel(t); ETo_t = ETo_serie(t);
    if isnan(P_t), P_t = 0; end
    if isnan(ETo_t), ETo_t = 0; end
    
    % 1) Inferencia predictiva del modelo Grey-Box
    x_pred_model = full(f_greybox(x_real, u_k, [P_t; ETo_t], t));
    
    % 2) Avance de la Planta Real (Simulación acoplada con perturbación real)
    x_real_next = rk4_step_real(f_real_ode, x_real, u_k, [P_t; ETo_t], t);
    
    % 3) Evolución libre y divergente de los estimadores nominales vs híbridos
    x_nom_next = full(f_nominal(x_nom_state, u_k, [P_t; ETo_t], t));
    x_pred_next  = full(f_greybox(x_pred_model, u_k, [P_t; ETo_t], t));
    
    % Proyecciones de saturación por límites biofísicos estrictos
    x_real_next(1) = min(max(x_real_next(1), 0.05), 0.50); x_real_next(2) = max(x_real_next(2), 0);
    x_pred_next(1) = min(max(x_pred_next(1), 0.05), 0.50); x_pred_next(2) = max(x_pred_next(2), 0);
    x_nom_next(1)  = min(max(x_nom_next(1), 0.05), 0.50);  x_nom_next(2)  = max(x_nom_next(2), 0);
    
    % Registro histórico indexado
    theta_real(t) = x_real_next(1); B_real(t)     = x_real_next(2); u_hist(t)     = u_k;
    theta_pred(t) = x_pred_next(1); B_pred(t)     = x_pred_next(2);
    theta_nom(t)  = x_nom_next(1);  B_nom(t)      = x_nom_next(2);
    
    % Avance recursivo del estado de la simulación
    x_real      = x_real_next;  
    x_nom_state = x_nom_next;   
    x_pred_model   = x_pred_next; 
end

%% 9. BLOQUE GRÁFICO DE ALTA RESOLUCIÓN PARA MONOGRAFÍA
dias = (1:Tsim)';
fs = 12; fs_tit = 14; black = [0.1 0.1 0.1];

% --- FIGURA 1: COMPORTAMIENTO DINÁMICO EN LAZO CERRADO ---
figure('Color','w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], 'Name', 'Desempeño NMPC Grey-Box');

% Subplot 1A: Dinámica del Contenido Volumétrico de Humedad
subplot(3,1,1)
plot(dias, theta_real, 'b', 'LineWidth', 2.5); hold on;
plot(dias, theta_pred, 'r--', 'LineWidth', 2);
plot(dias, theta_nom, 'k:', 'LineWidth', 1.5);
yline(theta_fc, '-.', 'CC', 'Color', [0.2 0.2 0.5], 'LineWidth', 1.2, 'FontSize', fs-2, 'FontWeight', 'bold', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top');
yline(theta_ref, '--', 'Ref', 'Color', [1 0 1], 'LineWidth', 1.5, 'FontSize', fs-2, 'FontWeight', 'bold', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top');
yline(theta_pwp, '-.', 'PMP', 'Color', [0.6 0 0], 'LineWidth', 1.2, 'FontSize', fs-2, 'FontWeight', 'bold', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top');
ylabel('\theta [m^3/m^3]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Control de Humedad del Suelo: Arquitectura NMPC Grey-Box', 'FontSize', fs_tit, 'FontWeight', 'bold','Color', black);
grid on;
xlim([1, Tsim]); % <--- CORRECCIÓN ANALÍTICA: Elimina el vacío al ceñirse al límite de días simulados
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);
lgd1 = legend('Planta Real', 'Modelo Grey-Box', 'Modelo Nominal', 'Location', 'northeastoutside');
set(lgd1, 'Color', 'w', 'EdgeColor', black, 'TextColor', black);

% Subplot 1B: Curvas de Acumulación Vegetativa de Biomasa
subplot(3,1,2)
plot(dias, B_real, 'b', 'LineWidth', 2.5); hold on;
plot(dias, B_pred, 'r--', 'LineWidth', 2);
plot(dias, B_nom, 'k:', 'LineWidth', 1.5);
yline(Bmax, 'g--', 'B_{max}', 'LineWidth', 1.5, 'FontSize', fs-2, 'FontWeight', 'bold', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom', 'Color', [0 0.8 0]);
ylabel('Biomasa [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Crecimiento Acumulado de Biomasa', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on;
xlim([1, Tsim]); % <--- CORRECCIÓN ANALÍTICA: Ajuste estricto del eje temporal
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);
lgd2 = legend('Planta Real', 'Modelo Grey-Box', 'Modelo Nominal', 'Location', 'northeastoutside');
set(lgd2, 'Color', 'w', 'EdgeColor', black, 'TextColor', black);

% Subplot 1C: Histograma de Acción de Control de Entrada (Lámina de Riego)
subplot(3,1,3)
stairs(dias, u_hist, 'k', 'LineWidth', 2.5); hold on;
yline(Imax, 'r:', 'Límite Riego', 'LineWidth', 2, 'FontSize', fs-2, 'FontWeight', 'bold');
ylabel('Riego [mm/día]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
xlabel('Tiempo [días]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Señal de Control NMPC (Acción de Riego Optimizada)', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
ylim([-0.5, Imax+2]); xlim([1, Tsim]); % <--- CORRECCIÓN ANALÍTICA
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);

% --- FIGURA 2: EVOLUCIÓN HISTÓRICA DE RESIDUOS DE PREDICCIÓN ---
figure('Color','w', 'Units', 'normalized', 'Position', [0.2 0.2 0.6 0.6], 'Name', 'Análisis de Errores');
subplot(2,1,1)
plot(dias, theta_real - theta_pred, 'r', 'LineWidth', 2); yline(0, 'k--', 'LineWidth', 1.2);
ylabel('Error \theta', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Residuo de Predicción Acumulado: Humedad', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on; xlim([1, Tsim]); % <--- CORRECCIÓN ANALÍTICA
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);

subplot(2,1,2)
plot(dias, B_real - B_pred, 'b', 'LineWidth', 2); yline(0, 'k--', 'LineWidth', 1.2);
ylabel('Error Biomasa [g/m^2]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
xlabel('Tiempo [días]', 'FontSize', fs, 'FontWeight', 'bold', 'Color', black);
title('Residuo de Predicción Acumulado: Biomasa', 'FontSize', fs_tit, 'FontWeight', 'bold', 'Color', black);
grid on; xlim([1, Tsim]); % <--- CORRECCIÓN ANALÍTICA
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.2);
set(gcf, 'PaperPositionMode', 'auto');

%% 10. EVALUACIÓN ESTADÍSTICA DE MÉTRICAS DE DESEMPEÑO COMPARATIVO
% Estimación de errores e indicadores para el predictor Caja Gris Híbrido
mae_theta = mean(abs(theta_real - theta_pred));
rmse_theta = sqrt(mean((theta_real - theta_pred).^2));
mae_biomasa = mean(abs(B_real - B_pred));
rmse_biomasa = sqrt(mean((B_real - B_pred).^2));

% --- MODELADO EN BUCLE DE LA SIMULACIÓN DE RIEGO CONVENCIONAL TRADICIONAL ---
u_tradicional = zeros(Tsim, 1);
theta_trad = 0.30; 
lamina_fija = 8.5;  
umbral_riego = 0.33; 

for t = 1:Tsim
    if theta_trad < umbral_riego
        u_t = lamina_fija; 
    else
        u_t = 0;
    end
    u_tradicional(t) = u_t;
    
    % Simulación de pérdidas por ineficiencia física en aspersión tradicional (~70% de eficiencia)
    riego_efectivo = u_t * 0.70; 
    ETc_t = 1.05 * 1.1 * ETo_serie(t); 
    theta_trad = theta_trad + (riego_efectivo + P_excel(t) - ETc_t)/(1000 * Zr);
end

% --- ANÁLISIS DE EFICIENCIA EN EL USO DEL AGUA (WUE) ---
consumo_nmpc = sum(u_hist);
consumo_total_trad = sum(u_tradicional);
ahorro_agua = ((consumo_total_trad - consumo_nmpc) / consumo_total_trad) * 100;

% Conversión física de índices de productividad de biomasa de g/m2 a kg/m3
biomasa_final_kg_m2 = B_real(end) / 1000; 
volumen_agua_m3_nmpc = consumo_nmpc / 1000; 
volumen_agua_m3_trad = consumo_total_trad / 1000;
WUE_nmpc = biomasa_final_kg_m2 / volumen_agua_m3_nmpc;
WUE_trad = biomasa_final_kg_m2 / volumen_agua_m3_trad;

% Determinación del indicador agronómico de estrés por déficit hídrico
limite_estres = theta_pwp + (1 - p_agotamiento)*(theta_fc - theta_pwp);
dias_estres = sum(theta_real < limite_estres);

%% 11. DECLARACIÓN DE RESULTADOS CONSOLIDADOS PARA TABLAS DE TRABAJO DE GRADO
fprintf('\n==========================================================\n');
fprintf('   RESULTADOS DE VALIDACIÓN: NMPC GREY-BOX VS TRADICIONAL   \n');
fprintf('==========================================================\n');
fprintf('PRECISIÓN DEL MODELO HÍBRIDO:\n');
fprintf('MAE Humedad: %.4f | RMSE Humedad: %.4f\n', mae_theta, rmse_theta);
fprintf('MAE Biomasa: %.2f g/m2 | RMSE Biomasa: %.2f g/m2\n', mae_biomasa, rmse_biomasa);
fprintf('\nDESEMPEÑO HÍDRICO (Ref: CCB / Guerrero-Guio):\n');
fprintf('  Consumo NMPC:        %.2f mm\n', consumo_nmpc);
fprintf('  Consumo Tradicional: %.2f mm\n', consumo_total_trad);
fprintf('  AHORRO DE AGUA:      %.2f %%\n', ahorro_agua);
fprintf('\nPRODUCTIVIDAD AGRO-EFICIENTE (WUE):\n');
fprintf('  WUE NMPC:            %.2f kg/m3\n', WUE_nmpc);
fprintf('  WUE Tradicional:     %.2f kg/m3\n', WUE_trad);
fprintf('  INCREMENTO WUE:      %.2f kg/m3 (Ref local: +3.49)\n', WUE_nmpc - WUE_trad);
fprintf('\nCONVENIENCIA AGRONÓMICA REGIONAL:\n');
fprintf('  Días de Estrés:      %d de %d días totales\n', dias_estres, Tsim);
fprintf('==========================================================\n');

% Exportación y guardado de resultados consolidados en formatos estructurados
resultado = table( ...
    fechas(1:Tsim), ...
    theta_real, theta_pred, theta_nom, ...
    B_real, B_pred, B_nom, ...
    u_hist, ...
    'VariableNames', {'Fecha','Theta_Real','Theta_GreyBox','Theta_Nominal','Biomasa_Real','Biomasa_GreyBox','Biomasa_Nominal','Riego_mm'});
writetable(resultado, 'Validacion_GreyBox_NMPC_Papa.xlsx');
save('validacion_greybox_nmpc_papa.mat', 'resultado', 'theta_real', 'theta_pred', 'theta_nom', 'B_real', 'B_pred', 'B_nom', 'u_hist');
disp('Validación NMPC Grey-Box completada. Archivos ".mat" y ".xlsx" generados con éxito.');

%% =========================================================================
%                  12. FUNCIONES DE SOPORTE MATEMÁTICO Y FÍSICO
% =========================================================================

%% --- MODELADO DE LA PLANTA REAL PERTURBADA (ODE) ---
function dx = planta_real_ode(t, x, u, p, theta_fc, theta_pwp, theta_c, Bmax, Zr, Kc_ini, Kc_mid, Kc_fin, Kd, alpha)
    theta = x(1); B = x(2); P = p(1); ETo = p(2);
    
    if B < 0.9 * Bmax
        Kc = Kc_ini + (Kc_mid - Kc_ini) * (B / Bmax);
    else
        Kc = Kc_mid - (Kc_mid - Kc_fin) * (1 - B / Bmax);
    end
    Kc = 1.03 * Kc; % Desviación paramétrica real del coeficiente del cultivo (+3%)
    
    Ks = max(0, min(1, (theta - theta_pwp) / (theta_c - theta_pwp)));
    ETc = 1.05 * Kc * Ks * ETo; % Variación e incertidumbre física en consumo evaporativo (+5%)
    D_vol = (Kd / alpha) * log(1 + exp(alpha * (theta - theta_fc)));
    
    delta_s = 0.008 + 0.022 / (1 + exp(-0.2 * (t - 105)));
    
    dtheta = (u + P - ETc) / (1000 * Zr) - D_vol;
    dB = (0.125 * Ks) * B * (1 - B / Bmax) - delta_s * B;
    dx = [dtheta; dB];
end

%% --- STEP DE INTEGRACIÓN NUMÉRICA RK4 (PLANTA REAL) ---
function x_next = rk4_step_real(f_ode, x, u, p, t)
    h = 1;
    k1 = f_ode(t, x, u, p);
    k2 = f_ode(t + h/2, x + h/2 * k1, u, p);
    k3 = f_ode(t + h/2, x + h/2 * k2, u, p);
    k4 = f_ode(t + h,   x + h   * k3, u, p);
    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

%% --- MÓDULO PSICROMETRÍA (FAO-56) ---
function [Tm, g, D] = modulo_psicrometria(Tmax, Tmin, alt)
    Tm = (Tmax + Tmin) / 2;
    Pres = 101.3 * ((293 - 0.0065 * alt) / 293)^5.26;
    g = 0.000665 * Pres;
    D = (4098 * (0.6108 * exp((17.27 * Tm) / (Tm + 237.3)))) / ((Tm + 237.3)^2);
end

%% --- MÓDULO PRESIÓN DE VAPOR (FAO-56) ---
function [es, ea] = modulo_vapor(Tmax, Tmin, HR)
    eTmax = 0.6108 * exp((17.27 * Tmax) / (Tmax + 237.3));
    eTmin = 0.6108 * exp((17.27 * Tmin) / (Tmin + 237.3));
    es = (eTmax + eTmin) / 2;
    ea = (HR / 100) * es;
end

%% --- MÓDULO RADIACIÓN SOLAR (FAO-56) ---
function Rn = modulo_radiacion(n, J, lat_deg, Tmax, Tmin, ea)
    lat_rad = lat_deg * (pi/180);
    dr = 1 + 0.033 * cos(2 * pi * J / 365);
    delta_s = 0.409 * sin(2 * pi * J / 365 - 1.39);
    ws = acos(-tan(lat_rad) * tan(delta_s));
    Ra = (24 * 60 / pi) * 0.0820 * dr * ...
        (ws * sin(lat_rad) * sin(delta_s) + cos(lat_rad) * cos(delta_s) * sin(ws));
    N = max((24 / pi) * ws, 1e-6);
    Rs = (0.25 + 0.50 * (n / N)) * Ra;
    Rns = (1 - 0.23) * Rs;
    Rso = (0.75 + 2e-5 * 2550) * Ra;
    sigma = 4.903e-9;
    TmaxK = Tmax + 273.16; TminK = Tmin + 273.16;
    Rnl = sigma * ((TmaxK^4 + TminK^4) / 2) * ...
        (0.34 - 0.14 * sqrt(max(ea, 0))) * ...
        (1.35 * (Rs / max(Rso, 1e-6)) - 0.35);
    Rn = Rns - Rnl;
end

%% --- GENERACIÓN DE LLUVIA CONTINUA SUAVE ---
function P_eff = calcular_lluvia_suave(P_max, t_actual, k, t_start, t_end)
    t_rel = t_actual - floor(t_actual); 
    sig1 = 1 / (1 + exp(-k * (t_rel - t_start)));
    sig2 = 1 / (1 + exp(-k * (t_rel - t_end)));
    P_eff = P_max * (sig1 - sig2);
end