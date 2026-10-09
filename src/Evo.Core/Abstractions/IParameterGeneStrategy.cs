namespace Evo.Core;

/// <summary>
/// Creation and mutation of parameter genes for the value domain <typeparamref name="T"/>.
/// Implementations must be stateless; all randomness comes from the supplied stream.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IParameterGeneStrategy<T>
{
    /// <summary>Creates an initial parameter gene value.</summary>
    T Create(ref RandomStream random);

    /// <summary>Returns a mutated parameter gene value.</summary>
    T Mutate(T current, ref RandomStream random);
}
