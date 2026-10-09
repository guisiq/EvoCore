namespace Evo.Core.Compiled;

/// <summary>
/// Bounded cache of compiled plans. Entries are evicted in insertion order when the capacity is reached.
/// </summary>
internal sealed class CompiledPlanCache<T>
{
    private readonly int _capacity;
    private readonly Dictionary<string, CompiledGenomePlan<T>> _plans = new(StringComparer.Ordinal);
    private readonly Queue<string> _order = new();
    private readonly object _gate = new();

    internal CompiledPlanCache(int capacity)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Cache capacity must be > 0.");
        }

        _capacity = capacity;
    }

    internal int Count
    {
        get
        {
            lock (_gate)
            {
                return _plans.Count;
            }
        }
    }

    internal CompiledGenomePlan<T> GetOrAdd(string key, Func<CompiledGenomePlan<T>> factory, out bool hit)
    {
        lock (_gate)
        {
            if (_plans.TryGetValue(key, out var existing))
            {
                hit = true;
                return existing;
            }

            hit = false;
            var plan = factory();
            if (_plans.Count >= _capacity)
            {
                _plans.Remove(_order.Dequeue());
            }

            _plans.Add(key, plan);
            _order.Enqueue(key);
            return plan;
        }
    }
}

/// <summary>
/// Compiled evaluator: compiles the active closure of a genome into a typed plan and reuses it while the
/// phenotype fingerprint, function set id and value type id stay the same. Supports all four official
/// value domains. Its output must equal <see cref="ReferenceGenomeEvaluator{T}"/>.
/// </summary>
/// <remarks>
/// Instances hold no per-call state; the value slots are allocated per call, so values never leak across calls.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public sealed class CompiledGenomeEvaluator<T> : IGenomeEvaluator<T>
{
    private readonly IFunctionSet<T> _functions;
    private readonly IValueSemantics<T> _values;
    private readonly CompiledPlanCache<T> _cache;
    private readonly CompiledExecutionCounters? _counters;
    private long _hits;
    private long _misses;

    /// <summary>Creates a compiled evaluator.</summary>
    /// <param name="functions">Function set used to resolve function genes.</param>
    /// <param name="values">Value semantics used to clone parameters and build cache keys.</param>
    /// <param name="cacheCapacity">Maximum number of cached plans (bounded cache).</param>
    /// <param name="counters">Opt-in execution counters; null in timing runs.</param>
    public CompiledGenomeEvaluator(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int cacheCapacity = 256,
        CompiledExecutionCounters? counters = null)
    {
        ArgumentNullException.ThrowIfNull(functions);
        ArgumentNullException.ThrowIfNull(values);
        _functions = functions;
        _values = values;
        _cache = new CompiledPlanCache<T>(cacheCapacity);
        _counters = counters;
    }

    /// <summary>Number of plans currently cached.</summary>
    public int CacheCount => _cache.Count;

    /// <summary>Number of evaluations that reused a cached plan.</summary>
    public long CacheHits => Interlocked.Read(ref _hits);

    /// <summary>Number of evaluations that compiled a new plan.</summary>
    public long CacheMisses => Interlocked.Read(ref _misses);

    /// <inheritdoc />
    public void Evaluate(Genome<T> genome, ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        ArgumentNullException.ThrowIfNull(genome);
        var shape = genome.Shape;
        if (inputs.Length != shape.InputCount)
        {
            throw new ArgumentException(
                $"Expected {shape.InputCount} inputs, got {inputs.Length}.", nameof(inputs));
        }

        if (outputs.Length != shape.OutputCount)
        {
            throw new ArgumentException(
                $"Expected {shape.OutputCount} output slots, got {outputs.Length}.", nameof(outputs));
        }

        var plan = GetPlan(genome);
        _counters?.Prepare(shape.NodeCount);
        plan.RunScalar(inputs, outputs, _counters?.ScalarExecutions);
    }

    /// <summary>Returns the cached plan of <paramref name="genome"/>, compiling it on a miss.</summary>
    internal CompiledGenomePlan<T> GetPlan(Genome<T> genome)
    {
        var fingerprint = genome.ComputePhenotypeFingerprint(_functions, _values);
        var key = $"{Convert.ToBase64String(fingerprint)}|{_functions.Id}|{_values.ValueTypeId}";
        var plan = _cache.GetOrAdd(key, () => CompiledGenomePlan<T>.Compile(genome, _functions, _values), out var hit);
        if (hit)
        {
            Interlocked.Increment(ref _hits);
        }
        else
        {
            Interlocked.Increment(ref _misses);
        }

        return plan;
    }
}
