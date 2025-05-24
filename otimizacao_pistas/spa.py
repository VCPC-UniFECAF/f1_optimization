import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle
from IPython.display import HTML

# Teste importar Json e plotagem
with open('circuitos.json', 'r') as f:
    data = json.load(f)
pontos = np.array(data['Spa'])
x = pontos[:, 0]
y = -pontos[:, 1]

# Parametros
t = np.linspace(0, 1, len(pontos))
cs_x = CubicSpline(t, pontos[:, 0], bc_type='periodic')
cs_y = CubicSpline(t, -pontos[:, 1], bc_type='periodic')

# Suavizar os pontos
t_smooth = np.linspace(0, 1, 500)
x_center = cs_x(t_smooth)
y_center = cs_y(t_smooth)

# Calculo de vetores para limites da pista
dx = cs_x.derivative(1)(t_smooth)
dy = cs_y.derivative(1)(t_smooth)
normals = np.array([-dy,dx]).T
normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

# Definição de largura da pista
width = 6
x_inner = x_center - width * normals[:, 0]
y_inner = y_center - width * normals[:, 1]
x_outer = x_center + width * normals[:, 0]
y_outer = y_center + width * normals[:, 1]

plt.figure(figsize=(20, 12))
plt.plot(x_center ,y_center, 'k-', label="Circuito Original", alpha = 1)
plt.plot(x_inner, y_inner, 'red', label='limite interno')
plt.plot(x_outer, y_outer, 'red', label='limite externo')
plt.title("Circuito de Spa")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()

def calc_comprimento():
    comprimento = 0
    for i in range(1, len(t_smooth)):
        dx = x_center[i] - x_center[i-1]
        dy = y_center[i] - y_center[i-1]
        comprimento += np.sqrt(dx**2 + dy**2)
    return comprimento

comp_atual = calc_comprimento()
escala = 7004 / comp_atual
pontos *= escala

cs_x = CubicSpline(t, pontos[:, 0], bc_type='periodic')
cs_y = CubicSpline(t, -pontos[:, 1], bc_type='periodic')

x_center = cs_x(t_smooth)
y_center = cs_y(t_smooth)
dx = cs_x.derivative(1)(t_smooth)
dy = cs_y.derivative(1)(t_smooth)
normals = np.array([-dy,dx]).T
normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

x_inner = x_center - width * normals[:, 0]
y_inner = y_center - width * normals[:, 1]
x_outer = x_center + width * normals[:, 0]
y_outer = y_center + width * normals[:, 1]

compr_final = calc_comprimento()
mu = 1.7             # Coeficiente de atrito
g = 9.81             # gravidade m/s²
mass = 950           # kg
power = 950 * 745.7  # converção HP para Watts
drag_coef = 0.7      # coeficiente de arrasto
frontal_area = 1.5   # m²
rho = 1.225          # densidade do ar
a = 13.89            # aceleração m/s²
dece = 33.3          # desaceleração m/s² 
mu_ef = dece / g     # coef. atrito
downforce160 =  750  # kd downforce a 160

# Calculos/funções físicos
def max_speed_in_curve(radius):
    return np.sqrt(mu * g * radius)

def max_acceleration(v):
    if v < 1e-6:
        return min(power / mass, mu * g)
    drag = 0.5 * rho * drag_coef * frontal_area * v**2
    available_power = min(power, power * (v/50))
    return (available_power - drag) / (mass * v)

def downforce(v_ms):
    return downforce160 * g * (v_ms / (160/3.6))**2

def max_dece(v_ms):
    df = downforce(v_ms)
    return mu_ef * (g + df / mass)

def calc_distancia(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

#Calculo raio de curvatura (curvas)
ddx = cs_x.derivative(2)(t_smooth)
ddy = cs_y.derivative(2)(t_smooth)
curvature = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
radius = np.abs(1/curvature)

# Simulação de volta
n_points = len(t_smooth)
velocities = np.zeros(n_points)
velocities[0] = 0 #velocidade inicial igual a 0 (alteral em m/s²)
distancias = np.zeros(n_points)

for i in range(1, n_points):
    distancias[i] = calc_distancia(x_center[i-1], y_center[i-1], x_center[i], y_center[i]) 

for i in range(1, n_points):
    v_max_curve = max_speed_in_curve(radius[i])

    if radius[i] > 300:    #Verificar se é reta ou curva, raio > 300 = reta
        a = max_acceleration(velocities[i-1])
        velocities[i] = min(velocities[i-1] + a * 0.1, 83.33)

    else:        #Frenagem progressiva para curva
        if velocities[i-1] > v_max_curve:
            dece = max_dece(velocities[i-1])
            dt = distancias[1] / velocities[i-1] if velocities[i-1] > 0 else 0.1
            velocities[i] = max(velocities[i-1] - dece * dt, v_max_curve)
        else:
            velocities[i] = v_max_curve

def objective(offsets):   #Calcula tempo total da volta
    total_time = 0
    prev_v = velocities [0]

    for i in range(n_points):   #Posição na pista
        offset = offsets[i]
        x = x_center[i] + offset * normals[i, 0]
        y = y_center[i] + offset * normals[i, 1]

        if i > 0:  #Distância entre pontos
            ds = calc_distancia(x_prev, y_prev, x, y)
            total_time += ds / ((prev_v + velocities[i]) / 2)    #tempo médio

        x_prev, y_prev = x, y
        prev_v = velocities[i]
    return total_time

bounds = [(-width, width) for _ in range(n_points)]  #Restrição para se manter na pista

#Otimização sugerida, só confiei
initial_offsets = np.zeros(n_points)
result = minimize(objective, initial_offsets, bounds=bounds, method='SLSQP')
optimal_offsets = result.x

def calcular_tempo_volta(velocidade, distancias):
    tempo = 0
    for i in range(1, len(velocidade)):
        if velocidade[i] > 0:
            tempo += distancias[i] / velocidade[i]
    return tempo

tempo_total = calcular_tempo_volta(velocities, distancias)

tempo_min = int(tempo_total // 60)
tempo_seg = int(tempo_total % 60)
tempo = f'{tempo_min}:{tempo_seg:02d}'

## Visualização de resultados
# Melhor trajetória
x_optimal = x_center + optimal_offsets * normals[:, 0]
y_optimal = y_center + optimal_offsets * normals[:, 1]

# Trajetória média
medium_offsets = optimal_offsets * 0.25
x_medium = x_center + medium_offsets * normals[:, 0]
y_medium = y_center + medium_offsets * normals[:, 1]

#Trajetória péssima
bad_offsets = np.where(optimal_offsets > 10, width*0.9, -width*0.9)
x_bad = x_center + bad_offsets * normals[:, 0]
y_bad = y_center + bad_offsets * normals[:, 0]

##Plot
plt.figure(figsize=(20, 12))
plt.plot(x_inner, y_inner, 'r-', alpha=0.3, label='limites da pista')
plt.plot(x_outer, y_outer, 'r-', alpha=0.3)
plt.plot(x_optimal, y_optimal, 'green', 'b-',  linewidth=2, label='trajetória ótima')
plt.plot(x_medium, y_medium, 'b-', linewidth=2, label='trajetória média')
plt.plot(x_bad, y_bad, 'red', 'b-', linewidth=2, label='trajetória ruim')
plt.text(
    0.3, 0.95, 
    f"Tempo total: {tempo}min",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8)
)
plt.legend()
plt.gca().set_aspect('equal')
plt.title('comparativo de rotas')
plt.show()
plt.figure(figsize=(20,12))
plt.plot(x_center, y_center, 'k--', alpha=0.3, label=f'Interlagos ({compr_final/1000:.3f}km)')
plt.plot(x_optimal,y_optimal, 'black', 'b-', linewidth=2, label='melhor trajetória')
plt.plot(x_inner, y_inner, 'r-', alpha=0.3, label='limites da pista')
plt.plot(x_outer, y_outer, 'r-', alpha=0.3)
plt.scatter(x_optimal[::1], y_optimal[::1], c=velocities[::1]*3.6,cmap='jet', s=30)
plt.colorbar(label='velocidade (km/h)')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()
tempo_acumulado = np.zeros(len(x_optimal))
for i in range(1, len(tempo_acumulado)):
    ds = np.sqrt((x_optimal[i]-x_optimal[i-1])**2 + (y_optimal[i]-y_optimal[i-1])**2)
    tempo_acumulado[i] = tempo_acumulado[i-1] + (ds / velocities[i] if velocities[i] > 0 else 0)

tempo_total = tempo_acumulado[-1]  # Tempo total da volta em segundos

# Configuração da animação
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
plt.title(f'Simulação Realista - Tempo Total: {int(tempo_total//60)}:{int(tempo_total%60):02d}')

# Circuito
ax.plot(x_center, y_center, 'k--', alpha=0.3)
ax.plot(x_inner, y_inner, 'gray', alpha=0.5)
ax.plot(x_outer, y_outer, 'gray', alpha=0.5)

# Elementos animados
bolinha = Circle((0, 0), radius=8, color='red', zorder=10)
ax.add_patch(bolinha)

info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Função de animação com tempo real
def animate(i):
    # Tempo atual proporcional ao tempo total
    tempo_atual = (i / frames_total) * tempo_total
    
    # Encontra o índice mais próximo do tempo atual
    idx = np.argmin(np.abs(tempo_acumulado - tempo_atual))
    
    # Atualiza posição e cor
    bolinha.center = (x_optimal[idx], y_optimal[idx])
    velocidade_kmh = velocities[idx] * 3.6
    cor = plt.cm.jet((velocidade_kmh - np.min(velocities*3.6)) / 
                   (np.max(velocities*3.6) - np.min(velocities*3.6)))
    bolinha.set_color(cor)
    
    # Atualiza texto
    info_text.set_text(
        f"Tempo: {int(tempo_atual//60)}:{int(tempo_atual%60):02d}\n"
        f"Velocidade: {velocidade_kmh:.1f} km/h\n"
        f"Setor: {int(idx/len(x_optimal)*3 + 1)}/3"
    )
    
    return bolinha, info_text

# 5. Configuração final da animação
frames_total = 150  # Número de frames para toda a volta
duracao_desejada = tempo_total  # Em segundos
intervalo = (duracao_desejada / frames_total) * 1000  # ms por frame

ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    frames=frames_total,
    interval=intervalo,
    blit=True
)

plt.close()
HTML(ani.to_jshtml())

# 1. Configuração da Imagem do Carro (versão compatível)
def carrega_carro(path, zoom=0.1):
    try:
        img = plt.imread(path)
    except FileNotFoundError:
        # Fallback: cria um carro simples
        img = np.zeros((50, 100, 4))  # Imagem transparente
        img[20:30, :] = [1, 0, 0, 1]  # Faixa vermelha central
    return OffsetImage(img, zoom=zoom)

# 2. Criação do objeto do carro
carro_img = carrega_carro('carrof1.png')
carro_artist = AnnotationBbox(carro_img, (0, 0), frameon=False)
ax.add_artist(carro_artist)

# Variável para armazenar a transformação atual
transformacao = Affine2D()

# 3. Função de Animação Corrigida
def animate(i):
    global transformacao
    
    tempo_atual = (i / frames_total) * tempo_total
    idx = np.argmin(np.abs(tempo_acumulado - tempo_atual))
    
    # Calcula ângulo de rotação (90° de ajuste para orientação da imagem)
    dx = x_optimal[(idx+1)%len(x_optimal)] - x_optimal[idx]
    dy = y_optimal[(idx+1)%len(y_optimal)] - y_optimal[idx]
    angulo = np.degrees(np.arctan2(dy, dx)) + 90
    
    # Atualiza posição
    carro_artist.xy = (x_optimal[idx], y_optimal[idx])
    
    # Aplica rotação corretamente
    transformacao = Affine2D().rotate_deg(angulo)
    carro_img.set_transform(transformacao + ax.transData)
    
    # Atualiza informações
    info_text.set_text(f"Tempo: {int(tempo_atual//60)}:{int(tempo_atual%60):02d}\n"
                      f"Velocidade: {velocities[idx]*3.6:.1f} km/h")   
    return carro_artist, info_text

# Configuração inicial
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(min(x_optimal)-50, max(x_optimal)+50)
ax.set_ylim(min(y_optimal)-50, max(y_optimal)+50)
ax.set_aspect('equal')

# Elementos estáticos
ax.plot(x_center, y_center, 'k--', alpha=0.3)
ax.plot(x_inner, y_inner, 'gray', alpha=0.5)
ax.plot(x_outer, y_outer, 'gray', alpha=0.5)

# Texto de informações
info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.rcParams['animation.embed_limit'] = 50

# Animação
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    frames=frames_total,
    interval=intervalo,
    blit=True
)

plt.close()
HTML(ani.to_jshtml())

# 1. Configuração Inicial
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_aspect('equal')
ax.set_xlim(min(x_center)-50, max(x_center)+50)
ax.set_ylim(min(y_center)-50, max(y_center)+50)

# 2. Carrega a imagem do carro (com fallback)
try:
    img = plt.imread('carrof1.png')
except FileNotFoundError:
    img = np.zeros((50, 100, 4))
    img[20:30, :] = [1, 0, 0, 1]  # Cria um retângulo vermelho como fallback

# 3. Cria o objeto do carro
carro_img = OffsetImage(img, zoom=0.01)
carro_artist = AnnotationBbox(carro_img, (x_optimal[0], y_optimal[0]), frameon=False)
ax.add_artist(carro_artist)

# 4. Função de Animação Corrigida
def animate(i):
    # Atualiza índice considerando o comprimento do array
    idx = i % len(x_optimal)
    
    # Calcula direção do movimento
    next_idx = (idx + 1) % len(x_optimal)
    dx = x_optimal[next_idx] - x_optimal[idx]
    dy = y_optimal[next_idx] - y_optimal[idx]
    
    # Calcula ângulo de rotação (em graus)
    angulo = np.degrees(np.arctan2(dy, dx)) + 90  # +90 para ajustar a orientação
    
    # Atualiza posição
    carro_artist.xybox = (x_optimal[idx], y_optimal[idx])
    
    # Aplica rotação
    trans = Affine2D().rotate_deg_around(x_optimal[idx], y_optimal[idx], angulo)
    carro_img.set_transform(trans + ax.transData)
    
    # Atualiza informações
    tempo_atual = tempo_acumulado[idx]
    info_text.set_text(f"Tempo: {int(tempo_atual//60)}:{int(tempo_atual%60):02d}\n"
                      f"Velocidade: {velocities[idx]*3.6:.1f} km/h")
    
    return carro_artist, info_text

# 5. Elementos do Circuito
ax.plot(x_center, y_center, 'k--', alpha=0.3, label='Linha central')
ax.plot(x_inner, y_inner, 'gray', alpha=0.5)
ax.plot(x_outer, y_outer, 'gray', alpha=0.5)
info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 6. Configuração da Animação
frames_total = len(x_optimal)  # Um frame para cada ponto
plt.rcParams['animation.embed_limit'] = 100
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    frames=frames_total,
    interval=20,  # Intervalo entre frames em ms
    blit=True,
    repeat=True
)

plt.close()
HTML(ani.to_jshtml())

# Configura a figura
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_aspect('equal')
ax.set_xlim(min(x_optimal)-50, max(x_optimal)+50)
ax.set_ylim(min(y_optimal)-50, max(y_optimal)+50)

# Plota o circuito
ax.plot(x_center, y_center, 'k--', alpha=0.3, label='Linha central')
ax.plot(x_inner, y_inner, 'gray', alpha=0.5)
ax.plot(x_outer, y_outer, 'gray', alpha=0.5)

# Carrega a imagem do carro (ou cria um fallback)
try:
    img_carro = plt.imread('carrof1.png')
except FileNotFoundError:
    img_carro = np.zeros((50, 100, 4))  # Imagem transparente
    img_carro[20:30, :] = [1, 0, 0, 1]  # Retângulo vermelho

# Cria o objeto do carro
carro_img = OffsetImage(img_carro, zoom=0.08)
carro_artist = AnnotationBbox(carro_img, (x_optimal[0], y_optimal[0]), frameon=False)
ax.add_artist(carro_artist)

# Texto de informações
info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes,fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Função de animação
def animate(i):
    idx = i % len(x_optimal)
    
    # Calcula direção do movimento
    next_idx = (idx + 1) % len(x_optimal)
    dx = x_optimal[next_idx] - x_optimal[idx]
    dy = y_optimal[next_idx] - y_optimal[idx]
    angulo = np.degrees(np.arctan2(dy, dx)) + 90  # +90 para alinhar a imagem
    
    # Atualiza posição e rotação
    carro_artist.xybox = (x_optimal[idx], y_optimal[idx])
    trans = Affine2D().rotate_deg_around(x_optimal[idx], y_optimal[idx], angulo)
    carro_img.set_transform(trans + ax.transData)
    
    # Atualiza informações
    tempo_atual = tempo_acumulado[idx]
    minutos = int(tempo_atual // 60)
    segundos = int(tempo_atual % 60)
    info_text.set_text(f"Tempo: {minutos}:{segundos:02d}\nVelocidade: {velocities[idx]*3.6:.1f} km/h")
    
    return carro_artist, info_text

# Configura a animação
frames_total = len(x_optimal)
ani = animation.FuncAnimation(
    fig=fig,
    func=animate,
    frames=frames_total,
    interval=20,  # 20ms por frame (~50 FPS)
    blit=True,
    repeat=True
)

plt.close()
HTML(ani.to_jshtml())