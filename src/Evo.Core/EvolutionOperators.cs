namespace Evo.Core;

/// <summary>
/// Operadores evolutivos v1: inicialização, torneio, crossover uniforme e mutação (02 §6).
/// </summary>
/// <remarks>
/// Slots de gene, em ordem: função (um por nó), conexão (um por entrada de conexão),
/// parâmetro (um por nó, arrays contam como um gene) e saída. Cada operação usa apenas o
/// <see cref="RandomStream"/> recebido, então o resultado depende da chave semântica do trabalho.
/// </remarks>
public sealed class EvolutionOperators<T>
{
    private readonly IGeneticProblem<T> _problem;
    private readonly GenomeShape _shape;
    private readonly int[] _functionIds;
    private readonly int _slotCount;

    /// <summary>Cria os operadores para o problema, validando que existe função compatível com MaxArity.</summary>
    public EvolutionOperators(IGeneticProblem<T> problem)
    {
        ArgumentNullException.ThrowIfNull(problem);
        _problem = problem;
        _shape = problem.Shape;

        var ids = new List<int>();
        for (var id = 0; id < problem.Functions.Count; id++)
        {
            if (problem.Functions.Get(id).Arity <= _shape.MaxArity)
            {
                ids.Add(id);
            }
        }

        if (ids.Count == 0)
        {
            throw new ArgumentException("Function set has no function with arity <= MaxArity.", nameof(problem));
        }

        _functionIds = ids.ToArray();
        _slotCount = (_shape.NodeCount * (_shape.MaxArity + 2)) + _shape.OutputCount;
    }

    /// <summary>Número total de slots de gene G.</summary>
    public int SlotCount => _slotCount;

    /// <summary>Compara fitness: válida sempre vence inválida; entre válidas, maior <c>Value</c> vence.</summary>
    public static int CompareFitness(FitnessResult left, FitnessResult right)
    {
        if (left.IsValid != right.IsValid)
        {
            return left.IsValid ? 1 : -1;
        }

        return left.Value.CompareTo(right.Value);
    }

    /// <summary>Cria um indivíduo inicial válido.</summary>
    public Genome<T> CreateInitial(ref RandomStream random)
    {
        var n = _shape.NodeCount;
        var m = _shape.MaxArity;
        var functionGenes = new int[n];
        var connectionGenes = new int[n * m];
        var parameterGenes = new T[n];
        var outputGenes = new int[_shape.OutputCount];

        for (var node = 0; node < n; node++)
        {
            functionGenes[node] = _functionIds[random.NextInt(0, _functionIds.Length)];
        }

        for (var node = 0; node < n; node++)
        {
            for (var slot = 0; slot < m; slot++)
            {
                connectionGenes[(node * m) + slot] = RandomConnection(node, ref random);
            }
        }

        for (var node = 0; node < n; node++)
        {
            parameterGenes[node] = _problem.Parameters.Create(ref random);
        }

        for (var output = 0; output < outputGenes.Length; output++)
        {
            outputGenes[output] = random.NextInt(0, _shape.InputCount + n);
        }

        return Build(functionGenes, connectionGenes, parameterGenes, outputGenes);
    }

    /// <summary>
    /// Gera um descendente: torneio para o pai A; com <c>CrossoverRate</c>, torneio para o pai B e
    /// crossover uniforme; depois mutação. O resultado é validado e recalculado.
    /// </summary>
    public Genome<T> Breed(
        Genome<T>[] population,
        FitnessResult[] fitness,
        EvolutionOptions options,
        ref RandomStream random)
    {
        return BreedWith(
            new TournamentParentSelector<T>(options.TournamentSize),
            population,
            fitness,
            options,
            ref random);
    }

    /// <summary>
    /// Mesma sequência de <see cref="Breed"/>, com a seleção de pais delegada a <paramref name="selector"/>.
    /// Índices fora da população são rejeitados antes de qualquer descendente ser construído.
    /// </summary>
    public Genome<T> BreedWith(
        IParentSelector<T> selector,
        IReadOnlyList<Genome<T>> population,
        IReadOnlyList<FitnessResult> fitness,
        EvolutionOptions options,
        ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(selector);
        ArgumentNullException.ThrowIfNull(options);
        var context = new ParentSelectionContext<T>(population, fitness);

        var a = population[SelectValidIndex(selector, context, ref random)];
        var functionGenes = a.FunctionGenes.ToArray();
        var connectionGenes = a.ConnectionGenes.ToArray();
        var parameterGenes = a.ParameterGenes.ToArray();
        var outputGenes = a.OutputGenes.ToArray();

        if (random.NextBernoulli(options.CrossoverRate))
        {
            var b = population[SelectValidIndex(selector, context, ref random)];
            Crossover(b, options.UniformCrossoverGeneProbability, ref random, functionGenes, connectionGenes, parameterGenes, outputGenes);
        }

        Mutate(functionGenes, connectionGenes, parameterGenes, outputGenes, ref random);
        return Build(functionGenes, connectionGenes, parameterGenes, outputGenes);
    }

    private static int SelectValidIndex(IParentSelector<T> selector, ParentSelectionContext<T> context, ref RandomStream random)
    {
        var index = selector.SelectParentIndex(context, ref random);
        if ((uint)index >= (uint)context.Count)
        {
            throw new InvalidOperationException(
                $"Parent selector returned index {index}, outside the population of size {context.Count}.");
        }

        return index;
    }

    private void Crossover(
        Genome<T> b,
        double geneProbability,
        ref RandomStream random,
        int[] functionGenes,
        int[] connectionGenes,
        T[] parameterGenes,
        int[] outputGenes)
    {
        var bf = b.FunctionGenes;
        for (var i = 0; i < functionGenes.Length; i++)
        {
            if (random.NextBernoulli(geneProbability))
            {
                functionGenes[i] = bf[i];
            }
        }

        var bc = b.ConnectionGenes;
        for (var i = 0; i < connectionGenes.Length; i++)
        {
            if (random.NextBernoulli(geneProbability))
            {
                connectionGenes[i] = bc[i];
            }
        }

        var bp = b.ParameterGenes;
        for (var i = 0; i < parameterGenes.Length; i++)
        {
            if (random.NextBernoulli(geneProbability))
            {
                parameterGenes[i] = _problem.Values.Clone(bp[i]);
            }
        }

        var bo = b.OutputGenes;
        for (var i = 0; i < outputGenes.Length; i++)
        {
            if (random.NextBernoulli(geneProbability))
            {
                outputGenes[i] = bo[i];
            }
        }
    }

    private void Mutate(
        int[] functionGenes,
        int[] connectionGenes,
        T[] parameterGenes,
        int[] outputGenes,
        ref RandomStream random)
    {
        var rate = 1.0 / _slotCount;
        var changed = false;
        for (var slot = 0; slot < _slotCount; slot++)
        {
            if (random.NextBernoulli(rate))
            {
                changed |= MutateSlot(slot, functionGenes, connectionGenes, parameterGenes, outputGenes, ref random);
            }
        }

        for (var attempt = 0; !changed && attempt < _slotCount * 4; attempt++)
        {
            var slot = random.NextInt(0, _slotCount);
            changed = MutateSlot(slot, functionGenes, connectionGenes, parameterGenes, outputGenes, ref random);
        }
    }

    private bool MutateSlot(
        int slot,
        int[] functionGenes,
        int[] connectionGenes,
        T[] parameterGenes,
        int[] outputGenes,
        ref RandomStream random)
    {
        var n = _shape.NodeCount;
        var m = _shape.MaxArity;
        if (slot < n)
        {
            return MutateFunction(slot, functionGenes, ref random);
        }

        slot -= n;
        if (slot < n * m)
        {
            return MutateConnection(slot, connectionGenes, ref random);
        }

        slot -= n * m;
        if (slot < n)
        {
            return MutateParameter(slot, parameterGenes, ref random);
        }

        return MutateOutput(slot - n, outputGenes, ref random);
    }

    private bool MutateFunction(int node, int[] functionGenes, ref RandomStream random)
    {
        var current = functionGenes[node];
        var len = _functionIds.Length;
        var index = random.NextInt(0, len);
        if (_functionIds[index] == current)
        {
            if (len == 1)
            {
                return false;
            }

            index = (index + 1) % len;
        }

        functionGenes[node] = _functionIds[index];
        return true;
    }

    private bool MutateConnection(int gene, int[] connectionGenes, ref RandomStream random)
    {
        var node = gene / _shape.MaxArity;
        var count = ConnectionCount(node);
        if (count == 1)
        {
            return false;
        }

        var current = connectionGenes[gene];
        var k = random.NextInt(0, count);
        if (ConnectionAt(node, k) == current)
        {
            k = (k + 1) % count;
        }

        connectionGenes[gene] = ConnectionAt(node, k);
        return true;
    }

    private bool MutateParameter(int node, T[] parameterGenes, ref RandomStream random)
    {
        var previous = parameterGenes[node];
        parameterGenes[node] = _problem.Parameters.Mutate(previous, ref random);
        return !_problem.Values.ValueEquals(previous, parameterGenes[node]);
    }

    private bool MutateOutput(int output, int[] outputGenes, ref RandomStream random)
    {
        var count = _shape.InputCount + _shape.NodeCount;
        if (count == 1)
        {
            return false;
        }

        var current = outputGenes[output];
        var k = random.NextInt(0, count);
        if (k == current)
        {
            k = (k + 1) % count;
        }

        outputGenes[output] = k;
        return true;
    }

    private int RandomConnection(int node, ref RandomStream random)
    {
        return ConnectionAt(node, random.NextInt(0, ConnectionCount(node)));
    }

    /// <summary>Número de endereços válidos para uma conexão do nó: entradas mais nós dentro de LevelsBack.</summary>
    private int ConnectionCount(int node)
    {
        return _shape.InputCount + (node - WindowStart(node));
    }

    /// <summary>Mapeia o índice k (0 até ConnectionCount) para o endereço correspondente.</summary>
    private int ConnectionAt(int node, int k)
    {
        return k < _shape.InputCount ? k : WindowStart(node) + k;
    }

    private int WindowStart(int node) => Math.Max(0, node - _shape.LevelsBack);

    private Genome<T> Build(int[] functionGenes, int[] connectionGenes, T[] parameterGenes, int[] outputGenes)
    {
        return Genome<T>.FromGenes(
            _shape,
            functionGenes,
            connectionGenes,
            parameterGenes,
            outputGenes,
            _problem.Values,
            _problem.Functions);
    }
}
