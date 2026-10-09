using System.Numerics;

namespace Evo.Core.Compiled;

/// <summary>Backend that actually executed a multi-case call.</summary>
public enum EvaluationBackend
{
    /// <summary>Compiled scalar plan (fallback or no full vector block).</summary>
    CompiledScalar,

    /// <summary>SIMD kernels over full vector blocks, with a scalar tail.</summary>
    Simd,
}

/// <summary>
/// Report of one multi-case call: the backend actually used and, on fallback, the explicit reason.
/// </summary>
/// <param name="Backend">Backend that executed the call.</param>
/// <param name="FallbackReason">Null when the SIMD path was used; otherwise why it was not.</param>
/// <param name="VectorWidth"><c>Vector&lt;T&gt;.Count</c> of the running hardware.</param>
/// <param name="CaseCount">Number of cases evaluated.</param>
/// <param name="VectorBlocks">Number of full vector blocks processed by SIMD kernels.</param>
/// <param name="TailCases">Cases processed by the scalar path.</param>
public sealed record EvaluationBackendReport(
    EvaluationBackend Backend,
    string? FallbackReason,
    int VectorWidth,
    int CaseCount,
    int VectorBlocks,
    int TailCases);

/// <summary>
/// SIMD evaluator for <c>double</c> and <c>int</c>. SIMD runs <em>inside one individual across several
/// cases</em> (not across individuals). Each active node is computed once per vector block; the
/// remaining cases use the scalar plan. If any active function lacks a kernel, the whole call falls back
/// explicitly to the compiled scalar path and the report says so.
/// </summary>
/// <typeparam name="T">Value type; only <c>double</c> and <c>int</c> have SIMD kernels.</typeparam>
public sealed class SimdGenomeEvaluator<T> where T : struct
{
    private readonly CompiledGenomeEvaluator<T> _compiled;
    private readonly CompiledExecutionCounters? _counters;

    /// <summary>Creates a SIMD evaluator.</summary>
    /// <param name="functions">Function set (must be the standard set of <c>T</c> to be SIMD-eligible).</param>
    /// <param name="values">Value semantics of <c>T</c>.</param>
    /// <param name="cacheCapacity">Bounded plan cache capacity.</param>
    /// <param name="counters">Opt-in execution counters; null in timing runs.</param>
    public SimdGenomeEvaluator(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int cacheCapacity = 256,
        CompiledExecutionCounters? counters = null)
    {
        _compiled = new CompiledGenomeEvaluator<T>(functions, values, cacheCapacity, counters);
        _counters = counters;
    }

    /// <summary>
    /// Evaluates <paramref name="caseCount"/> cases. <paramref name="inputCases"/> is case-major
    /// (<c>caseCount * InputCount</c>) and <paramref name="outputCases"/> is case-major (<c>caseCount * OutputCount</c>).
    /// </summary>
    /// <returns>The backend that actually ran and its diagnostics.</returns>
    public EvaluationBackendReport Evaluate(Genome<T> genome, ReadOnlySpan<T> inputCases, int caseCount, Span<T> outputCases)
    {
        ArgumentNullException.ThrowIfNull(genome);
        if (caseCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(caseCount), caseCount, "Case count must be >= 0.");
        }

        var shape = genome.Shape;
        if (inputCases.Length != caseCount * shape.InputCount)
        {
            throw new ArgumentException(
                $"Expected {caseCount * shape.InputCount} input values for {caseCount} cases, got {inputCases.Length}.",
                nameof(inputCases));
        }

        if (outputCases.Length != caseCount * shape.OutputCount)
        {
            throw new ArgumentException(
                $"Expected {caseCount * shape.OutputCount} output values for {caseCount} cases, got {outputCases.Length}.",
                nameof(outputCases));
        }

        var plan = _compiled.GetPlan(genome);
        var width = Vector<T>.Count;
        _counters?.Prepare(shape.NodeCount);

        if (!plan.SimdEligible)
        {
            RunScalarCases(plan, inputCases, caseCount, outputCases);
            return new EvaluationBackendReport(EvaluationBackend.CompiledScalar, plan.SimdFallbackReason, width, caseCount, 0, caseCount);
        }

        var blocks = caseCount / width;
        if (blocks == 0)
        {
            RunScalarCases(plan, inputCases, caseCount, outputCases);
            return new EvaluationBackendReport(
                EvaluationBackend.CompiledScalar,
                $"caseCount {caseCount} menor que a largura vetorial {width}; sem bloco SIMD.",
                width,
                caseCount,
                0,
                caseCount);
        }

        var inputCount = shape.InputCount;
        var outputCount = shape.OutputCount;

        // Transposed buffers: value of input i for case c lives at [i * caseCount + c].
        var inputT = new T[inputCount * caseCount];
        for (var c = 0; c < caseCount; c++)
        {
            for (var i = 0; i < inputCount; i++)
            {
                inputT[i * caseCount + c] = inputCases[c * inputCount + i];
            }
        }

        var outputT = new T[outputCount * caseCount];
        RunVectorBlocks(plan, inputT, caseCount, blocks, outputT);

        var tailStart = blocks * width;
        var tail = caseCount - tailStart;
        if (tail > 0)
        {
            var scratch = new T[outputCount];
            for (var c = tailStart; c < caseCount; c++)
            {
                plan.RunScalar(inputCases.Slice(c * inputCount, inputCount), scratch, _counters?.TailExecutions);
                for (var o = 0; o < outputCount; o++)
                {
                    outputT[o * caseCount + c] = scratch[o];
                }
            }
        }

        for (var c = 0; c < caseCount; c++)
        {
            for (var o = 0; o < outputCount; o++)
            {
                outputCases[c * outputCount + o] = outputT[o * caseCount + c];
            }
        }

        return new EvaluationBackendReport(EvaluationBackend.Simd, null, width, caseCount, blocks, tail);
    }

    private void RunScalarCases(CompiledGenomePlan<T> plan, ReadOnlySpan<T> inputCases, int caseCount, Span<T> outputCases)
    {
        var inputCount = plan.InputCount;
        var outputCount = plan.OutputCount;
        for (var c = 0; c < caseCount; c++)
        {
            plan.RunScalar(
                inputCases.Slice(c * inputCount, inputCount),
                outputCases.Slice(c * outputCount, outputCount),
                _counters?.ScalarExecutions);
        }
    }

    private void RunVectorBlocks(CompiledGenomePlan<T> plan, T[] inputT, int caseCount, int blocks, T[] outputT)
    {
        var width = Vector<T>.Count;
        var inputCount = plan.InputCount;
        var slots = new Vector<T>[plan.SlotCount];
        var nodes = plan.Nodes;
        var destinations = plan.Destinations;
        var ops = plan.SimdOps;
        var arguments = plan.Arguments;
        var parameters = plan.Parameters;
        var maxArity = plan.MaxArity;
        var outputs = plan.OutputAddresses;
        var counters = _counters;

        // Only int needs the explicit Abs(int.MinValue) == int.MaxValue rule of the scalar semantics.
        var isInt = typeof(T) == typeof(int);
        var intMin = isInt ? new Vector<T>((T)(object)int.MinValue) : default;
        var intMax = isInt ? new Vector<T>((T)(object)int.MaxValue) : default;

        for (var b = 0; b < blocks; b++)
        {
            var offset = b * width;
            for (var i = 0; i < inputCount; i++)
            {
                slots[i] = new Vector<T>(inputT, i * caseCount + offset);
            }

            for (var k = 0; k < nodes.Length; k++)
            {
                var baseIndex = k * maxArity;
                Vector<T> value = ops[k] switch
                {
                    SimdOp.Constant => new Vector<T>(parameters[k]),
                    SimdOp.Add => slots[arguments[baseIndex]] + slots[arguments[baseIndex + 1]],
                    SimdOp.Subtract => slots[arguments[baseIndex]] - slots[arguments[baseIndex + 1]],
                    SimdOp.Multiply => slots[arguments[baseIndex]] * slots[arguments[baseIndex + 1]],
                    SimdOp.Negate => -slots[arguments[baseIndex]],
                    SimdOp.Abs => AbsVector(slots[arguments[baseIndex]], isInt, intMin, intMax),
                    SimdOp.Square => SquareVector(slots[arguments[baseIndex]]),
                    SimdOp.Min => Vector.Min(slots[arguments[baseIndex]], slots[arguments[baseIndex + 1]]),
                    SimdOp.Max => Vector.Max(slots[arguments[baseIndex]], slots[arguments[baseIndex + 1]]),
                    _ => throw new InvalidOperationException($"Node {nodes[k]} has no SIMD kernel."),
                };
                slots[destinations[k]] = value;
                if (counters is not null)
                {
                    counters.KernelExecutions[nodes[k]]++;
                }
            }

            for (var o = 0; o < outputs.Length; o++)
            {
                slots[outputs[o]].CopyTo(outputT, o * caseCount + offset);
            }
        }
    }

    private static Vector<T> AbsVector(Vector<T> x, bool isInt, Vector<T> intMin, Vector<T> intMax)
    {
        var abs = Vector.Abs(x);
        return isInt ? Vector.ConditionalSelect(Vector.Equals(x, intMin), intMax, abs) : abs;
    }

    private static Vector<T> SquareVector(Vector<T> x) => x * x;
}
