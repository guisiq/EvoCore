using Xunit;

namespace Evo.Core.Tests;

public sealed class RandomStreamTests
{
    [Fact]
    public void Version_is_frozen()
    {
        Assert.Equal("rng/xoshiro256ss-sha256-key/v1", RandomStream.Version);
    }

    [Fact]
    public void Reference_vector_state_and_outputs_for_seed42_gen0_ind0_stream0()
    {
        // Normative vector: 02-dag-e-evolucao.md section 5.
        var stream = RandomStream.Create(42, 0, 0, 0);

        ulong[] expected =
        [
            13153161111426186266UL,
            924257236617452906UL,
            5351099750999899165UL,
            15344191101087260527UL,
            9674610500839565023UL,
            7279954407479924424UL,
            14975208716801538411UL,
            17685779050241604496UL,
        ];

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], stream.NextUInt64());
        }
    }

    [Fact]
    public void Same_key_yields_same_sequence_regardless_of_request_order()
    {
        var a1 = RandomStream.Create(7, 3, 11, 2);
        var b1 = RandomStream.Create(7, 3, 12, 2);
        var a2 = RandomStream.Create(7, 3, 11, 2);
        var b2 = RandomStream.Create(7, 3, 12, 2);

        // Interleaved order A then B.
        var seqA1 = new ulong[16];
        var seqB1 = new ulong[16];
        for (int i = 0; i < 16; i++)
        {
            seqA1[i] = a1.NextUInt64();
            seqB1[i] = b1.NextUInt64();
        }

        // Reverse order B then A.
        var seqB2 = new ulong[16];
        var seqA2 = new ulong[16];
        for (int i = 0; i < 16; i++)
        {
            seqB2[i] = b2.NextUInt64();
            seqA2[i] = a2.NextUInt64();
        }

        Assert.Equal(seqA1, seqA2);
        Assert.Equal(seqB1, seqB2);
        Assert.NotEqual(seqA1, seqB1);
    }

    [Fact]
    public void Simulated_serial_and_parallel_schedules_produce_identical_outputs()
    {
        const int individuals = 64;
        const int draws = 32;

        var serial = new ulong[individuals][];
        for (int ind = 0; ind < individuals; ind++)
        {
            var s = RandomStream.Create(99, 5, (ulong)ind, 1);
            serial[ind] = new ulong[draws];
            for (int d = 0; d < draws; d++)
            {
                serial[ind][d] = s.NextUInt64();
            }
        }

        // Simulated scheduling: reverse order with worker-like chunking.
        var parallel = new ulong[individuals][];
        for (int ind = individuals - 1; ind >= 0; ind--)
        {
            var s = RandomStream.Create(99, 5, (ulong)ind, 1);
            parallel[ind] = new ulong[draws];
            for (int d = 0; d < draws; d++)
            {
                parallel[ind][d] = s.NextUInt64();
            }
        }

        for (int ind = 0; ind < individuals; ind++)
        {
            Assert.Equal(serial[ind], parallel[ind]);
        }
    }

    [Fact]
    public void NextDouble_is_in_unit_interval_and_uses_53_bits()
    {
        var stream = RandomStream.Create(1, 0, 0, 0);
        var reference = RandomStream.Create(1, 0, 0, 0);
        for (int i = 0; i < 1000; i++)
        {
            double d = stream.NextDouble();
            ulong raw = reference.NextUInt64();
            Assert.InRange(d, 0.0, 1.0);
            Assert.True(d < 1.0);
            Assert.Equal((raw >> 11) * (1.0 / 9007199254740992.0), d);
        }
    }

    [Fact]
    public void NextSingle_is_in_unit_interval_and_uses_24_bits()
    {
        var stream = RandomStream.Create(1, 0, 0, 1);
        var reference = RandomStream.Create(1, 0, 0, 1);
        for (int i = 0; i < 1000; i++)
        {
            float f = stream.NextSingle();
            ulong raw = reference.NextUInt64();
            Assert.InRange(f, 0.0f, 1.0f);
            Assert.True(f < 1.0f);
            Assert.Equal((raw >> 40) * (1.0f / 16777216.0f), f);
        }
    }

    [Fact]
    public void NextInt_respects_bounds_and_uses_rejection_sampling()
    {
        var stream = RandomStream.Create(3, 1, 2, 3);
        var counts = new int[7];
        for (int i = 0; i < 70_000; i++)
        {
            int v = stream.NextInt(10, 17);
            Assert.InRange(v, 10, 16);
            counts[v - 10]++;
        }

        foreach (int c in counts)
        {
            Assert.InRange(c, 9_000, 11_000);
        }
    }

    [Fact]
    public void NextInt_rejects_empty_range()
    {
        var stream = RandomStream.Create(3, 1, 2, 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => stream.NextInt(5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => stream.NextInt(5, 4));
    }

    [Fact]
    public void NextInt_handles_full_int_range_without_overflow()
    {
        var stream = RandomStream.Create(4, 0, 0, 0);
        for (int i = 0; i < 1000; i++)
        {
            int v = stream.NextInt(int.MinValue, int.MaxValue);
            Assert.InRange((long)v, int.MinValue, int.MaxValue - 1L);
        }
    }

    [Fact]
    public void Bernoulli_uses_next_double_less_than_p()
    {
        var a = RandomStream.Create(5, 0, 0, 0);
        var b = RandomStream.Create(5, 0, 0, 0);
        for (int i = 0; i < 500; i++)
        {
            bool expected = b.NextDouble() < 0.3;
            Assert.Equal(expected, a.NextBernoulli(0.3));
        }
    }

    [Fact]
    public void Shuffle_is_permutation_and_deterministic()
    {
        int[] first = Enumerable.Range(0, 20).ToArray();
        int[] second = Enumerable.Range(0, 20).ToArray();

        var s1 = RandomStream.Create(8, 2, 0, 0);
        var s2 = RandomStream.Create(8, 2, 0, 0);
        s1.Shuffle<int>(first);
        s2.Shuffle<int>(second);

        Assert.Equal(first, second);
        Assert.Equal(Enumerable.Range(0, 20), first.OrderBy(x => x));
    }
}
