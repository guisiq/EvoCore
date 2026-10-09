namespace Evo.Core;

/// <summary>
/// CGP genome with contiguous gene vectors. The exact gene layout, DAG addressing,
/// active-graph analysis and phenotype fingerprint are normative in
/// 02-dag-e-evolucao.md and are implemented by EVO-002. EVO-001 only freezes the
/// public type seam used by the frozen contracts.
/// </summary>
/// <remarks>
/// Instances are owned by a single worker at a time. Internal gene arrays are
/// private to the genome; offspring creation and metadata recomputation are the
/// responsibility of the genetic operators.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public sealed class Genome<T>
{
    /// <summary>Creates a genome carrying the fixed shape.</summary>
    public Genome(GenomeShape shape) => Shape = shape;

    /// <summary>Fixed shape of this genome.</summary>
    public GenomeShape Shape { get; }
}
