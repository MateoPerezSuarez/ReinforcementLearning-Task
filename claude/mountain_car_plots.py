import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

STUDY_NAME = "mountain_car_tabular"
STORAGE = f"sqlite:///{STUDY_NAME}.db"

def generate_plots():
    """Genera las 5 gráficas principales de Optuna"""
    
    # Cargar estudio
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    
    # Crear directorio para gráficas
    plot_dir = "optuna_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generando gráficas de Optuna...")
    print("-" * 60)
    
    # 1. Optimization History (Historial de optimización)
    print("1. Generando Optimization History...")
    fig = optuna.visualization.plot_optimization_history(study).to_plotly_figure()
    fig.write_image(f"{plot_dir}/01_optimization_history.png", width=1200, height=700)
    print("   ✓ Guardado: 01_optimization_history.png")
    
    # 2. Parameter Importances (Importancia de parámetros)
    print("2. Generando Parameter Importances...")
    fig = optuna.visualization.plot_param_importances(study).to_plotly_figure()
    fig.write_image(f"{plot_dir}/02_param_importances.png", width=1200, height=700)
    print("   ✓ Guardado: 02_param_importances.png")
    
    # 3. Slice Plot (Relación parámetros-valor)
    print("3. Generando Slice Plot...")
    fig = optuna.visualization.plot_slice(study).to_plotly_figure()
    fig.write_image(f"{plot_dir}/03_slice_plot.png", width=1200, height=700)
    print("   ✓ Guardado: 03_slice_plot.png")
    
    # 4. Parallel Coordinates (Coordenadas paralelas)
    print("4. Generando Parallel Coordinates...")
    fig = optuna.visualization.plot_parallel_coordinate(study).to_plotly_figure()
    fig.write_image(f"{plot_dir}/04_parallel_coordinates.png", width=1200, height=700)
    print("   ✓ Guardado: 04_parallel_coordinates.png")
    
    # 5. Algorithm Comparison (Comparación de algoritmos)
    print("5. Generando Algorithm Comparison...")
    fig = plt.figure(figsize=(12, 7))
    
    trials_df = study.trials_dataframe()
    
    # Subplot: Box plot por algoritmo
    ax1 = plt.subplot(1, 2, 1)
    algorithms = trials_df['params_algorithm'].unique()
    data_by_algo = [trials_df[trials_df['params_algorithm'] == algo]['value'].values 
                    for algo in algorithms]
    
    bp = ax1.boxplot(data_by_algo, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Reward (Mean of last 10 episodes)', fontsize=11)
    ax1.set_title('Distribución de Rewards por Algoritmo', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot: Conteo de trials por algoritmo
    ax2 = plt.subplot(1, 2, 2)
    algo_counts = trials_df['params_algorithm'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax2.bar(algo_counts.index, algo_counts.values, color=colors[:len(algo_counts)])
    ax2.set_ylabel('Número de Trials', fontsize=11)
    ax2.set_title('Distribución de Trials por Algoritmo', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/05_algorithm_comparison.png", dpi=150, bbox_inches='tight')
    print("   ✓ Guardado: 05_algorithm_comparison.png")
    plt.close()
    
    # 6. Bonus: Learning Rate vs Gamma scatter (distribución de hiperparámetros)
    print("6. Generando Learning Rate vs Gamma (bonus)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(trials_df['params_alpha'], 
                        trials_df['params_gamma'], 
                        c=trials_df['value'],
                        s=200, 
                        alpha=0.6,
                        cmap='RdYlGn',
                        edgecolors='black',
                        linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reward', fontsize=11)
    ax.set_xlabel('Learning Rate (Alpha)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discount Factor (Gamma)', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate vs Discount Factor (coloreado por Reward)', 
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/06_lr_vs_gamma.png", dpi=150, bbox_inches='tight')
    print("   ✓ Guardado: 06_lr_vs_gamma.png")
    plt.close()
    
    print("-" * 60)
    print(f"✓ Todas las gráficas guardadas en: {plot_dir}/")
    print(f"\nResumen del estudio:")
    print(f"  - Total de trials: {len(study.trials)}")
    print(f"  - Mejor valor: {study.best_value:.4f}")
    print(f"  - Mejor trial: #{study.best_trial.number}")

if __name__ == "__main__":
    try:
        import plotly
        generate_plots()
    except ImportError:
        print("⚠ Necesitas instalar plotly y kaleido para exportar las gráficas:")
        print("  pip install plotly kaleido")