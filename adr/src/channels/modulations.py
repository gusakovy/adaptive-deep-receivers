import jax.numpy as jnp

MODULATIONS = {
    "BPSK": jnp.array([-1, 1], dtype=jnp.float32),
    "QPSK": jnp.array([[jnp.cos(jnp.pi * (k + 1 / 2) / 2) +
                        jnp.sin(jnp.pi * (k + 1 / 2) / 2) * 1j] for k in range(4)], dtype=jnp.complex64),
    "16-QAM": jnp.array([complex(x, y) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]], dtype=jnp.complex64) / jnp.sqrt(2)
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (mod_name, mod_values) in zip(axs, MODULATIONS.items()):
        ax.scatter(mod_values.real, mod_values.imag)
        ax.set_title(mod_name)
        ax.set_xlabel('In-phase')
        ax.set_ylabel('Quadrature')
        ax.grid(True)

    plt.tight_layout()
    plt.show(block=True)
