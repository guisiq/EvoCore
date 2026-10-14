using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Vetores golden derivados por scripts/golden-fingerprint.cs (oraculo independente que monta
// os bytes canonicos da especificacao 02 sem usar Genome<T>). Nao regenerar a partir do codigo sob teste.
public sealed class FingerprintGoldenTests
{
    private sealed class GoldenFunction<T> : IFunction<T>
    {
        public string Id => "add";
        public int Arity => 2;
        public T Invoke(ReadOnlySpan<T> arguments, T parameter) => throw new NotSupportedException();
    }

    private sealed class GoldenFunctionSet<T> : IFunctionSet<T>
    {
        private readonly GoldenFunction<T> _function = new();
        public string Id => "golden-set";
        public int Count => 1;
        public IFunction<T> Get(int functionId) => _function;
    }

    private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();

    private static string Fingerprint<T>(T parameter, IValueSemantics<T> values)
    {
        var genome = Genome<T>.FromGenes(
            new GenomeShape(1, 1, 1, 2, 1),
            [0],
            [0, 0],
            [parameter],
            [1],
            values,
            new GoldenFunctionSet<T>());
        return Hex(genome.ComputePhenotypeFingerprint(new GoldenFunctionSet<T>(), values));
    }

    [Fact]
    public void Golden_fingerprint_double()
    {
        Assert.Equal(
            "0a024b182cee67f1ccf8a52880c2b1bbf553c5a6c085aa55942983eba3e8cbfd",
            Fingerprint(1.5, new DoubleValueSemantics()));
    }

    [Fact]
    public void Golden_fingerprint_int()
    {
        Assert.Equal(
            "1d5396b96ea55db6fc5d31fc1b6b6577f750d377c00495d14814cc2fde7b0585",
            Fingerprint(7, new Int32ValueSemantics()));
    }

    [Fact]
    public void Golden_fingerprint_double_array()
    {
        Assert.Equal(
            "559e587110e99fc939b41f5d3fa0e6a6ec6ed4db5e11d6573d063c6795ab6595",
            Fingerprint(new[] { 1.5, -0.0 }, new DoubleArrayValueSemantics()));
    }

    [Fact]
    public void Golden_fingerprint_int_array()
    {
        Assert.Equal(
            "5a79850ec112c8adeea36051654da8b2f7a698d62d6a047df4b2eb763138f242",
            Fingerprint(new[] { 3, 4 }, new Int32ArrayValueSemantics()));
    }
}
