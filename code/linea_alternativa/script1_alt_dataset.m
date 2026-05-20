% =========================================================================
% PROYECTO: CONTROL HÍBRIDO NMPC + RED NEURONAL RESIDUAL (GREY-BOX)
% SCRIPT 1 (ALT): GENERACIÓN DE DATASET CON REFERENCIA DINÁMICA VIA NMPC
% LINEA: TRAYECTORIA VARIABLE INTERNACIONAL (COL -> ARG -> USA)
% OBJETIVO: Construir un espacio de datos representativo (MISO) forzando al
%           NMPC a operar en tres regímenes de humedad distintos para que
%           la ANN aprenda los residuos físicos del suelo en todo el espacio.
% =========================================================================

clc; clear; close all;
import casadi.*

%% 0. CONFIGURACIÓN DE RUTAS INSTITUCIONALES
casadi_path = 'C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b';
if ~exist(casadi_path, 'dir')
    error('Ruta de CasADi inválida. Verifique la dirección física del toolkit.');
end
addpath(genpath(casadi_path));

%% 1. PARÁMETROS BIOFÍSICOS GENERALES (CULTIVO DE PAPA)
altitud = 2550; latitud = 4.69; u2 = 2.0; G = 0;
theta_fc  = 0.36; theta_pwp = 0.18; p_agotamiento = 0.35; Zr = 0.40;
theta_c   = theta_fc - p_agotamiento * (theta_fc - theta_pwp);
Kc_ini = 0.45; Kc_mid = 1.15; Kc_fin = 0.75; Bmax = 1500;

%% 2. PARSEO DE SERIES HISTÓRICAS IDEAM Y EVAPOTRANSPIRACIÓN (FAO-56)
archivo = 'Datos_IDEAM_Tibaitata_2015_SEQUIA_AJUSTADA';
tabla   = readtable(archivo);

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
J_dias = day(fechas, 'dayofyear');
nTot = height(tabla);
ETo_serie = zeros(nTot, 1);

for t = 1:nTot
    [Tm, gam, delt] = modulo_psicrometria(Tmax(t), Tmin(t), altitud);
    [es, ea]        = modulo_vapor(Tmax(t), Tmin(t), HR(t));
    Rn              = modulo_radiacion(brillo(t), J_dias(t), latitud, Tmax(t), Tmin(t), ea);
    
    ETo_serie(t)    = (0.408*delt*Rn + gam*(900/(Tm+273))*u2*(es-ea)) / (delt + gam*(1+0.34*u2));
    if ~isfinite(ETo_serie(t)) || ETo_serie(t) < 0
        ETo_serie(t) = 0;
    end
end

%% 3. VECTOR DE REFERENCIA DINÁMICA
% Debe ser IDÉNTICO al que usa el Script 3 alternativo para consistencia.
% Escalones suavizados con rampas de 7 días entre regímenes.
Tsim     = height(tabla) - 10; % N=10 en este script (igual que Script 1 original)
ref_vals   = [0.32, 0.30, 0.28]; % Configuración experimental: USA → ARG → COL (de mayor a menor)
breaks     = [45, 90];
trans_days = 7;               % Días de transición (rampa lineal)
theta_ref_vector = zeros(1, Tsim);

for k = 1:Tsim
    if k <= breaks(1) - trans_days
        theta_ref_vector(k) = ref_vals(1);                          % USA puro (0.32)
    elseif k <= breaks(1)
        alpha_t = (k - (breaks(1) - trans_days)) / trans_days;
        theta_ref_vector(k) = ref_vals(1) + alpha_t*(ref_vals(2) - ref_vals(1)); % rampa USA → ARG
    elseif k <= breaks(2) - trans_days
        theta_ref_vector(k) = ref_vals(2);                          % ARG puro (0.30)
    elseif k <= breaks(2)
        alpha_t = (k - (breaks(2) - trans_days)) / trans_days;
        theta_ref_vector(k) = ref_vals(2) + alpha_t*(ref_vals(3) - ref_vals(2)); % rampa ARG → COL
    else
        theta_ref_vector(k) = ref_vals(3);                          % COL puro (0.28)
    end
end

fprintf('Vector de referencia dinámica generado:\n');
fprintf('  Días 1-%d:     θ_ref = %.2f (USA)\n', breaks(1)-trans_days, ref_vals(1));
fprintf('  Días %d-%d:   rampa %.2f → %.2f (transición)\n', breaks(1)-trans_days+1, breaks(1), ref_vals(1), ref_vals(2));
fprintf('  Días %d-%d:  θ_ref = %.2f (ARG)\n', breaks(1)+1, breaks(2)-trans_days, ref_vals(2));
fprintf('  Días %d-%d:  rampa %.2f → %.2f (transición)\n', breaks(2)-trans_days+1, breaks(2), ref_vals(2), ref_vals(3));
fprintf('  Días %d-%d: θ_ref = %.2f (COL)\n', breaks(2)+1, Tsim, ref_vals(3));

%% 4. MODELADO FISIOLÓGICO NOMINAL SIMBÓLICO (CASADI)
x     = SX.sym('x', 2);   % Estados: [Humedad volumétrica; Biomasa]
u     = SX.sym('u');       % Control de entrada de riego
p     = SX.sym('p', 2);   % Perturbaciones del sistema: [Precipitación; ETo]
t_sym = SX.sym('t');       % Instante de tiempo

Kc_sym = if_else(x(2) < 0.9 * Bmax, ...
    Kc_ini + (Kc_mid - Kc_ini) * (x(2) / Bmax), ...
    Kc_mid - (Kc_mid - Kc_fin) * (1 - x(2) / Bmax));
Ks_sym = fmin(1, fmax(0, (x(1) - theta_pwp) / (theta_c - theta_pwp)));

ETc_sim    = Ks_sym * Kc_sym * p(2);
delta_s_nom = 0.01 + 0.015 / (1 + exp(-0.15 * (t_sym - 105)));
P_suave = calcular_lluvia_suave(p(1), t_sym, 30, 0.4, 0.7);

% Formulación del Espacio de Estados Continuo Nominal
rhs = [(1/(Zr*1000))*(u + P_suave - ETc_sim - (0.6/60)*log(1+exp(60*(x(1)-theta_fc)))*1000); ...
       (0.12 * Ks_sym) * x(2) * (1 - x(2)/Bmax) - delta_s_nom * x(2)];
f = Function('f', {x, u, p, t_sym}, {rhs});

% --- DIGITALIZACIÓN TEMPORAL POR INTEGRACIÓN RUNGE-KUTTA 4 ---
h  = 1;
k1 = f(x, u, p, t_sym);
k2 = f(x + h/2*k1, u, p, t_sym + h/2);
k3 = f(x + h/2*k2, u, p, t_sym + h/2);
k4 = f(x + h*k3,   u, p, t_sym + h);
f_rk4 = Function('f_rk4', {x, u, p, t_sym}, {x + h/6*(k1 + 2*k2 + 2*k3 + k4)});

%% 5. FORMULACIÓN DE LA FUNCIÓN DE COSTO EN EL PROGRAMA NO LINEAL (NLP)
N    = 10;  % Horizonte de predicción extendido para recolección de dinámicas
X    = SX.sym('X', 2, N+1);
U    = SX.sym('U', N);
X0   = SX.sym('X0', 2);
P_h  = SX.sym('P_h', N);
ETo_h = SX.sym('ETo_h', N);
t0   = SX.sym('t0');
theta_target = SX.sym('theta_target'); % Inyección paramétrica de setpoint dinámico

opt_vars = [reshape(U, N, 1); reshape(X, 2*(N+1), 1)];
Cost = 0; g = [X(:,1) - X0];

for k = 1:N
    % Ponderación de trayectoria alineada con la sintonización del Script de validación
    Cost = Cost + 5000*(X(1,k+1) - theta_target)^2 + 0.8*U(k)^2 - 0.01*X(2,k+1);
    g    = [g; X(:,k+1) - f_rk4(X(:,k), U(k), [P_h(k); ETo_h(k)], t0 + k - 1)];
end

opts = struct('ipopt', struct('print_level', 0, 'max_iter', 200, ...
              'tol', 1e-6, 'acceptable_tol', 1e-4), 'print_time', 0);
solver = nlpsol('solver', 'ipopt', struct('x', opt_vars, 'f', Cost, 'g', g, 'p', [X0; t0; theta_target; P_h; ETo_h]), opts);

%% 6. SIMULACIÓN CERRADA DE EXTRACCIÓN Y COLECCIÓN DE RESIDUOS
xk_real = [0.30; 5.0];
Xdata   = zeros(Tsim, 5);
Ydata   = zeros(Tsim, 2);
Imax    = 6;

lbw  = [zeros(N, 1);           repmat([theta_pwp; 0],    N+1, 1)];
ubw  = [Imax*ones(N, 1);       repmat([0.45; Bmax],      N+1, 1)];
w0   = zeros(length(lbw), 1);

fprintf('\nEjecutando lazo cerrado del NMPC para minería de datos residuo-físicos...\n');
for t = 1:Tsim
    current_ref = theta_ref_vector(t);
    p_input = [xk_real; t; current_ref; P_excel(t:t+N-1); ETo_serie(t:t+N-1)];
    
    try
        sol = solver('x0', w0, 'p', p_input, 'lbx', lbw, 'ubx', ubw, 'lbg', 0, 'ubg', 0);
        u_k = full(sol.x(1));
        u_k = min(max(u_k, 0), Imax);
        w0  = full(sol.x); 
    catch ME
        warning('Falla de convergencia en optimizador NLP del día t=%d. Se asume u=0. Error: %s', t, ME.message);
        u_k = 0;
    end
    
    p_t = [P_excel(t); ETo_serie(t)];
    
    % 1. Evolución del Proceso Real (Planta perturbada con desajuste paramétrico)
    x_next_real = rk4_step_real(@planta_real_ode, xk_real, u_k, p_t, t, ...
                  theta_fc, theta_pwp, theta_c, Bmax, Zr, Kc_ini, Kc_mid, Kc_fin);
              
    % 2. Evolución del Modelo Físico Nominal Puro
    x_next_nom = full(f_rk4(xk_real, u_k, p_t, t));
    
    % Mapeo de vectores del Dataset Híbrido
    Xdata(t, :) = [xk_real(1), xk_real(2), u_k, p_t(1), p_t(2)];
    Ydata(t, :) = (x_next_real - x_next_nom)';
    
    xk_real = x_next_real;
    
    if mod(t, 25) == 0
        fprintf('  Progreso [%d/%d] -> θ: %.3f | B: %.1f | u_riego: %.2f mm | Setpoint: %.2f\n', ...
                t, Tsim, xk_real(1), xk_real(2), u_k, current_ref);
    end
end

% Guardado explícito para no comprometer el dataset base de setpoint fijo
save('dataset_hibrido_refdin.mat', 'Xdata', 'Ydata', 'theta_ref_vector');
fprintf('\nExtracción exitosa. Dataset guardado bajo: "dataset_hibrido_refdin.mat"\n');

%% 7. ANÁLISIS DE INTEGRIDAD ESTADÍSTICA DEL DATASET EXTRÁIDO
fprintf('\n=============== DIAGNÓSTICO METROLÓGICO DEL DATASET ===============\n');
fprintf('Residuo de Humedad Suelo (θ): Media = %.6f | Desv. Est = %.6f | Abs Max = %.6f\n', ...
        mean(Ydata(:,1)), std(Ydata(:,1)), max(abs(Ydata(:,1))));
fprintf('Residuo de Biomasa Cultivo (B): Media = %.4f | Desv. Est = %.4f | Abs Max = %.4f\n', ...
        mean(Ydata(:,2)), std(Ydata(:,2)), max(abs(Ydata(:,2))));
fprintf('Lámina de Riego Media Distribuida: %.3f mm/día | Frecuencia de Cierre (Días u=0): %d\n', ...
        mean(Xdata(:,3)), sum(Xdata(:,3) < 1e-4));
fprintf('Límites Operacionales de Humedad: Media = %.4f | Mínima = %.4f | Máxima = %.4f\n', ...
        mean(Xdata(:,1)), min(Xdata(:,1)), max(Xdata(:,1)));
fprintf('====================================================================\n');

%% =========================================================================
%                         8. FUNCIONES DE TRANSPORTE
% =========================================================================

function dx = planta_real_ode(t, x, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf)
    if x(2) < 0.9 * Bmax
        Kc = Kci + (Kcm - Kci) * (x(2) / Bmax);
    else
        Kc = Kcm - (Kcm - Kcf) * (1 - x(2) / Bmax);
    end
    Ks  = max(0, min(1, (x(1) - twp) / (tc - twp)));
    ETc = (Kc * 1.05) * Ks * p(2); % Desajuste del 5% en consumo
    D   = (0.6/60) * log(1 + exp(60 * (x(1) - tfc))) * (1000 * Zr);
    delta_s = 0.008 + 0.022 / (1 + exp(-0.2 * (t - 105)));
    dB  = (0.125 * Ks) * x(2) * (1 - x(2)/Bmax) - delta_s * x(2);
    dx  = [(1/(Zr*1000))*(u + p(1) - ETc - D); dB];
end

function x_next = rk4_step_real(f_ode, x, u, p, t, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf)
    h  = 1;
    k1 = f_ode(t,     x,           u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k2 = f_ode(t+h/2, x+h/2*k1,   u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k3 = f_ode(t+h/2, x+h/2*k2,   u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k4 = f_ode(t+h,   x+h*k3,     u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

function [Tm, g, D] = modulo_psicrometria(Tmax, Tmin, alt)
    Tm   = (Tmax + Tmin)/2;
    Pres = 101.3 * ((293 - 0.0065*alt)/293)^5.26;
    g    = 0.000665 * Pres;
    D    = (4098 * (0.6108 * exp((17.27*Tm)/(Tm+237.3)))) / ((Tm+237.3)^2);
end

function [es, ea] = modulo_vapor(Tmax, Tmin, HR)
    eTmax = 0.6108 * exp((17.27*Tmax)/(Tmax+237.3));
    eTmin = 0.6108 * exp((17.27*Tmin)/(Tmin+237.3));
    es    = (eTmax + eTmin)/2;
    ea    = (HR/100)*es;
end

function Rn = modulo_radiacion(n, J, lat_deg, Tmax, Tmin, ea)
    lat_rad = lat_deg * (pi/180);
    dr      = 1 + 0.033 * cos(2*pi*J/365);
    delta_s = 0.409 * sin(2*pi*J/365 - 1.39);
    ws      = acos(-tan(lat_rad)*tan(delta_s));
    Ra      = (24*60/pi) * 0.0820 * dr * ...
              (ws*sin(lat_rad)*sin(delta_s) + cos(lat_rad)*cos(delta_s)*sin(ws));
    N_hrs   = max((24/pi)*ws, 1e-6);
    Rs      = (0.25 + 0.50*(n/N_hrs))*Ra;
    Rns     = (1 - 0.23)*Rs;
    Rso     = (0.75 + 2e-5*2550)*Ra;
    sig     = 4.903e-9;
    TmaxK   = Tmax + 273.16; TminK = Tmin + 273.16;
    Rnl     = sig * ((TmaxK^4 + TminK^4)/2) * ...
              (0.34 - 0.14*sqrt(max(ea,0))) * ...
              (1.35*(Rs/max(Rso,1e-6)) - 0.35);
    Rn = Rns - Rnl;
end

function P_eff = calcular_lluvia_suave(P_max, t_actual, k, t_start, t_end)
    t_rel = t_actual - floor(t_actual);
    sig1  = 1 / (1 + exp(-k * (t_rel - t_start)));
    sig2  = 1 / (1 + exp(-k * (t_rel - t_end)));
    P_eff = P_max * (sig1 - sig2);
end