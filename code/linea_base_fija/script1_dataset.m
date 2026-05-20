% =========================================================================
% PROYECTO: CONTROL HÍBRIDO NMPC + RED NEURONAL RESIDUAL (GREY-BOX)
% OBJETIVO: Generación de dataset sintético de entrenamiento (Residuos)
% MÉTODO: Integración RK4 para dinámica de 2 estados (Humedad y Biomasa)
% LINEA: REFERENCIA ESTÁTICA
% =========================================================================

addpath(genpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b'));
clc; clear; close all; import casadi.*

%% 1. PARÁMETROS FIJOS Y CONFIGURACIÓN AGRONÓMICA
% Parámetros físicos del suelo y entorno geográfico de la Sabana de Bogotá
altitud = 2550; latitud = 4.69; u2 = 2.0; G = 0;
theta_fc = 0.36; theta_pwp = 0.18; p_agotamiento = 0.35; Zr = 0.40;
theta_c = theta_fc - p_agotamiento * (theta_fc - theta_pwp);

% Dinámica biológica y coeficientes de desarrollo del cultivo de papa
Kc_ini = 0.45; Kc_mid = 1.15; Kc_fin = 0.75; Bmax = 1500;

%% 2. LECTURA DE DATOS Y CÁLCULO DE ETo (FAO-56)
% Conservación de escenarios de validación (Descomentar según la simulación deseada)
archivo = 'Datos_IDEAM_Tibaitata.xlsx'; %archivo original
%archivo = 'Datos_IDEAM_Tibaitata_2015_SEQUIA_AJUSTADA'; %sequia con datos alterados
%archivo = 'Datos_IDEAM_Tibaitata_LLUVIA.xlsx'; %archivo para escenario de lluvia

tabla = readtable(archivo);

% Extracción y conversión de registros meteorológicos de la tabla Excel
Tmax = str2double(string(tabla.Tmax)); 
Tmin = str2double(string(tabla.Tmin));
HR = str2double(string(tabla.HR)); 
brillo = str2double(string(tabla.Brillo));
P_excel = str2double(string(tabla.P));

% Normalización y validación del vector de fechas para el cálculo del día Juliano
if isdatetime(tabla.Fecha)
    fechas = tabla.Fecha; 
else
    fechas = datetime(tabla.Fecha); 
end
J_dias = day(fechas, 'dayofyear');

% Pre-cálculo iterativo de la serie diaria de Evapotranspiración de Referencia (ETo)
ETo_serie = zeros(height(tabla), 1);
for t = 1:height(tabla)
    [Tm, gam, delt] = modulo_psicrometria(Tmax(t), Tmin(t), altitud);
    [es, ea] = modulo_vapor(Tmax(t), Tmin(t), HR(t));
    Rn = modulo_radiacion(brillo(t), J_dias(t), latitud, Tmax(t), Tmin(t), ea);
    ETo_serie(t) = (0.408*delt*Rn + gam*(900/(Tm+273))*u2*(es-ea)) / (delt + gam*(1+0.34*u2));
end

%% 3. MODELO SIMBÓLICO Y SOLVER RK4 (CASADI)
x = SX.sym('x', 2);  % Vector de estados no lineales: [Humedad volumétrica; Biomasa acumulada]
u = SX.sym('u');     % Variable manipulada de control: Acción de Riego artificial
p = SX.sym('p', 2);  % Parámetros meteorológicos exógenos: [Lluvia del día; ETo calculada]
t_sym = SX.sym('t'); % Variable de tiempo continuo (Días del ciclo)

% Modelado simbólico de curvas fenológicas (Kc) acopladas al crecimiento de biomasa
Kc_sym = if_else(x(2) < 0.9 * Bmax, ...
    Kc_ini + (Kc_mid - Kc_ini) * (x(2) / Bmax), ...
    Kc_mid - (Kc_mid - Kc_fin) * (1 - x(2) / Bmax));

% Determinación del coeficiente de restricción hídrica (Ks) por agotamiento útil
Ks_sym = fmin(1, fmax(0, (x(1) - theta_pwp) / (theta_c - theta_pwp)));

% Cálculo de evapotranspiración real del cultivo (ETc) y coeficiente de senescencia
ETc_sim = Ks_sym * Kc_sym * p(2);
delta_s_nom = 0.01 + 0.015 / (1 + exp(-0.15 * (t_sym - 105)));

% Suavizado matemático de la precipitación diaria para asegurar la diferenciabilidad del solver NLP
P_suave = calcular_lluvia_suave(p(1), t_sym, 30, 0.4, 0.7);

% Definición del sistema de ecuaciones diferenciales continuas (RHS)
% Estado 1: Balance hídrico radicular con drenaje exponencial ante saturación (theta_fc)
% Estado 2: Ecuación logística de acumulación vegetativa penalizada por estrés hídrico
rhs = [(1/(Zr*1000))*(u + P_suave - ETc_sim - (0.6/60)*log(1+exp(60*(x(1)-theta_fc)))*1000); ...
       (0.12 * Ks_sym) * x(2) * (1 - x(2)/Bmax) - delta_s_nom * x(2)];
f = Function('f', {x, u, p, t_sym}, {rhs});

% Algoritmo de integración numérica Runge-Kutta de 4to orden (Simbólico)
h = 1;
k1 = f(x, u, p, t_sym);
k2 = f(x + h/2*k1, u, p, t_sym + h/2);
k3 = f(x + h/2*k2, u, p, t_sym + h/2);
k4 = f(x + h*k3, u, p, t_sym + h);
f_rk4 = Function('f_rk4', {x, u, p, t_sym}, {x + h/6*(k1 + 2*k2 + 2*k3 + k4)});

%% 4. CONFIGURACIÓN DEL CONTROLADOR OPTIMIZADOR NMPC
N = 10; % Horizonte de predicción temporal (7 días)
X = SX.sym('X', 2, N+1); 
U = SX.sym('U', N); 
X0 = SX.sym('X0', 2);
P_h = SX.sym('P_h', N); 
ETo_h = SX.sym('ETo_h', N); 
t0 = SX.sym('t0');

% Vector estructurado de variables de decisión para el optimizador no lineal
opt_vars = [reshape(U, N, 1); reshape(X, 2*(N+1), 1)]; 
Cost = 0; 
g = [X(:,1) - X0]; % Restricción de igualdad inicial (Inicialización del horizonte)

for k = 1:N
    % Funciones de costo preservadas para la alternación de escenarios del proyecto:
    Cost = Cost + 2500*(X(1,k+1) - 0.32)^2 + 0.05*U(k)^2 - 0.01*X(2,k+1); %Para el excel original
    %Cost = Cost + 1200*(X(1,k+1) - 0.32)^2 + 20.0*U(k)^2 - 0.001*X(2,k+1); %%Para el escenario de lluvia y sequia con 6 mm y 3.5mm
    %Cost = Cost + 5000*(X(1,k+1) - 0.30)^2 + 0.001*U(k)^2 - 0.01*X(2,k+1); %Para el escenario sequia 2 mm
    
    % Enlace dinámico de los estados futuros a través de las restricciones del integrador RK4
    g = [g; X(:,k+1) - f_rk4(X(:,k), U(k), [P_h(k); ETo_h(k)], t0 + k - 1)];
end

% Configuración estructural y supresión de visualizaciones del resolvedor IPOPT
opts = struct('ipopt', struct('print_level', 0), 'print_time', 0);
solver = nlpsol('solver', 'ipopt', struct('x', opt_vars, 'f', Cost, 'g', g, 'p', [X0; t0; P_h; ETo_h]), opts);

%% 5. SIMULACIÓN TEMPORAL Y GENERACIÓN DE DATASET HÍBRIDO
Tsim = height(tabla) - N;
xk_real = [0.30; 5.0]; % Condiciones iniciales iniciales de la planta real [Humedad; Biomasa]
Xdata = zeros(Tsim, 5); 
Ydata = zeros(Tsim, 2);

% Definición de cotas operativas (Límites inferiores y superiores de estados y control)
lbw = [zeros(N, 1); repmat([0.18; 0], N+1, 1)];
ubw = [6*ones(N, 1); repmat([0.45; 2000], N+1, 1)]; % Límite superior de riego variable según escenario
w0 = zeros(length(lbw), 1);

fprintf('Iniciando simulación de %d días...\n', Tsim);
for t = 1:Tsim
    % Organización del vector de parámetros del optimizador
    p_input = [xk_real; t; P_excel(t:t+N-1); ETo_serie(t:t+N-1)];
    
    % Ejecución del optimizador NLP (IPOPT)
    sol = solver('x0', w0, 'p', p_input, 'lbx', lbw, 'ubx', ubw, 'lbg', 0, 'ubg', 0);
    u_k = full(sol.x(1)); % Extracción de la primera acción de control (Horizonte Recesivo)
    p_t = [P_excel(t); ETo_serie(t)];
    
    % --- CÁLCULO DE RESIDUOS DE CAJA BLANCA ---
    % 1. Evolución de la Planta Real (Incertidumbre física y consumo hídrico alterado)
    x_next_real = rk4_step_real(@planta_real_ode, xk_real, u_k, p_t, t, ...
                  theta_fc, theta_pwp, theta_c, Bmax, Zr, Kc_ini, Kc_mid, Kc_fin);
    
    % 2. Evolución del Modelo Nominal (Estimación idealizada dentro del controlador)
    x_next_nom = full(f_rk4(xk_real, u_k, p_t, t));
    
    % Almacenamiento estructurado en matrices de características (Xdata) y objetivos (Ydata)
    Xdata(t, :) = [xk_real(1), xk_real(2), u_k, p_t(1), p_t(2)];
    Ydata(t, :) = (x_next_real - x_next_nom)';
    
    % Actualización recursiva de estados y semillas del optimizador
    xk_real = x_next_real; 
    w0 = full(sol.x);
end

% Almacenamiento del dataset en archivo binario de datos estructurados (.mat)
save('dataset_hibrido.mat', 'Xdata', 'Ydata');
disp('Dataset Híbrido generado correctamente y guardado como dataset_hibrido.mat.');

%% =========================================================================
%                  6. FUNCIONES DE SOPORTE MATEMÁTICO Y FÍSICO
% =========================================================================

%% --- MODELADO DE LA PLANTA REAL (ODE) ---
function dx = planta_real_ode(t, x, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf)
    % Representación del sistema dinámico real sujeto a variaciones de consumo y senescencia
    if x(2) < 0.9 * Bmax
        Kc = Kci + (Kcm - Kci) * (x(2) / Bmax);
    else
        Kc = Kcm - (Kcm - Kcf) * (1 - x(2) / Bmax);
    end
    
    Ks = max(0, min(1, (x(1) - twp) / (tc - twp)));
    ETc = (Kc * 1.05) * Ks * p(2); % Incertidumbre paramétrica física (+5% consumo)
    D = (0.6/60) * log(1 + exp(60 * (x(1) - tfc))) * (1000 * Zr);
    
    % Desviación biológica en la tasa logística de senescencia
    delta_s = 0.008 + 0.022 / (1 + exp(-0.2 * (t - 105)));
    dB = (0.125 * Ks) * x(2) * (1 - x(2)/Bmax) - delta_s * x(2);
    
    dx = [(1/(Zr*1000))*(u + p(1) - ETc - D); dB];
end

%% --- STEP DE INTEGRACIÓN NUMÉRICA RK4 (PLANTA REAL) ---
function x_next = rk4_step_real(f_ode, x, u, p, t, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf)
    h = 1;
    k1 = f_ode(t, x, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k2 = f_ode(t+h/2, x+h/2*k1, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k3 = f_ode(t+h/2, x+h/2*k2, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    k4 = f_ode(t+h, x+h*k3, u, p, tfc, twp, tc, Bmax, Zr, Kci, Kcm, Kcf);
    x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

%% --- MÓDULO PSICROMETRÍA (FAO-56) ---
function [Tm, g, D] = modulo_psicrometria(Tmax, Tmin, alt)
    Tm = (Tmax + Tmin)/2;
    Pres = 101.3 * ((293 - 0.0065*alt)/293)^5.26;
    g = 0.000665 * Pres;
    D = (4098 * (0.6108 * exp((17.27*Tm)/(Tm+237.3))))/((Tm+237.3)^2);
end

%% --- MÓDULO PRESIÓN DE VAPOR (FAO-56) ---
function [es, ea] = modulo_vapor(Tmax, Tmin, HR)
    eTmax = 0.6108 * exp((17.27*Tmax)/(Tmax+237.3));
    eTmin = 0.6108 * exp((17.27*Tmin)/(Tmin+237.3));
    es = (eTmax + eTmin)/2; 
    ea = (HR/100)*es;
end

%% --- MÓDULO RADIACIÓN SOLAR SOLAR (FAO-56) ---
function Rn = modulo_radiacion(n, J, lat_deg, Tmax, Tmin, ea)
    lat_rad = lat_deg * (pi/180); 
    dr = 1 + 0.033 * cos(2*pi*J/365);
    delta_s = 0.409 * sin(2*pi*J/365 - 1.39); 
    ws = acos(-tan(lat_rad)*tan(delta_s));
    Ra = (24*60/pi) * 0.0820 * dr * (ws*sin(lat_rad)*sin(delta_s) + cos(lat_rad)*cos(delta_s)*sin(ws));
    N = (24/pi)*ws; 
    Rs = (0.25+0.50*(n/N))*Ra; 
    Rns = (1-0.23)*Rs; 
    Rso = (0.75+2e-5*2550)*Ra;
    sig = 4.903e-9; 
    TmaxK = Tmax+273.16; TminK = Tmin+273.16;
    Rnl = sig * ((TmaxK^4 + TminK^4)/2) * (0.34 - 0.14*sqrt(ea)) * (1.35*(Rs/Rso) - 0.35);
    Rn = Rns - Rnl;
end

%% --- GENERACIÓN DE LLUVIA CONTINUA SUAVE ---
function P_eff = calcular_lluvia_suave(P_max, t_actual, k, t_start, t_end)
    t_rel = t_actual - floor(t_actual); 
    sig1 = 1 / (1 + exp(-k * (t_rel - t_start)));
    sig2 = 1 / (1 + exp(-k * (t_rel - t_end)));
    P_eff = P_max * (sig1 - sig2);
end