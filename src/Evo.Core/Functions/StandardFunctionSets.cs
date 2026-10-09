namespace Evo.Core.Functions;

/// <summary>Delegate body of a built-in function. Must not mutate its arguments.</summary>
internal delegate T FunctionBody<T>(ReadOnlySpan<T> arguments, T parameter);

/// <summary>Built-in node function backed by a pure delegate.</summary>
internal sealed class StandardFunction<T> : IFunction<T>
{
    private readonly FunctionBody<T> _body;

    internal StandardFunction(string id, int arity, FunctionBody<T> body)
    {
        Id = id;
        Arity = arity;
        _body = body;
    }

    /// <inheritdoc />
    public string Id { get; }

    /// <inheritdoc />
    public int Arity { get; }

    /// <inheritdoc />
    public T Invoke(ReadOnlySpan<T> arguments, T parameter) => _body(arguments, parameter);
}

/// <summary>Immutable ordered function set shared by the standard built-in sets.</summary>
internal sealed class StandardFunctionSet<T> : IFunctionSet<T>
{
    private readonly IFunction<T>[] _functions;

    internal StandardFunctionSet(string id, IFunction<T>[] functions)
    {
        Id = id;
        _functions = functions;
    }

    /// <inheritdoc />
    public string Id { get; }

    /// <inheritdoc />
    public int Count => _functions.Length;

    /// <inheritdoc />
    public IFunction<T> Get(int functionId)
    {
        if ((uint)functionId >= (uint)_functions.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(functionId),
                functionId,
                $"Function id must be in [0, {_functions.Length}) for function set '{Id}'.");
        }

        return _functions[functionId];
    }
}

/// <summary>Scalar semantics of the <c>double</c> built-ins (02 section 4).</summary>
internal static class DoubleOps
{
    /// <summary>Epsilon of the protected operations.</summary>
    internal const double Eps = 1e-12;

    internal static double Add(double a, double b) => a + b;

    internal static double Subtract(double a, double b) => a - b;

    internal static double Multiply(double a, double b) => a * b;

    internal static double ProtectedDivide(double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b))
        {
            return double.NaN;
        }

        return Math.Abs(b) <= Eps ? a : a / b;
    }

    internal static double Negate(double a) => -a;

    internal static double Abs(double a) => Math.Abs(a);

    internal static double Square(double a) => a * a;

    internal static double ProtectedSqrt(double x) =>
        double.IsNaN(x) ? double.NaN : Math.Sqrt(Math.Abs(x));

    internal static double ProtectedLog(double x) =>
        double.IsNaN(x) ? double.NaN : Math.Log(Math.Abs(x) + Eps);

    internal static double Min(double a, double b) => Math.Min(a, b);

    internal static double Max(double a, double b) => Math.Max(a, b);

    internal static double GreaterThan(double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b))
        {
            return double.NaN;
        }

        return a > b ? 1.0 : 0.0;
    }

    internal static double Select(double c, double a, double b) => c > 0 ? a : b;
}

/// <summary>Scalar semantics of the <c>int</c> built-ins (02 section 4).</summary>
internal static class Int32Ops
{
    internal static int Add(int a, int b) => unchecked(a + b);

    internal static int Subtract(int a, int b) => unchecked(a - b);

    internal static int Multiply(int a, int b) => unchecked(a * b);

    internal static int ProtectedDivide(int a, int b)
    {
        if (b == 0)
        {
            return a;
        }

        if (a == int.MinValue && b == -1)
        {
            return int.MaxValue;
        }

        return a / b;
    }

    internal static int Negate(int a) => unchecked(-a);

    internal static int Abs(int a) => a == int.MinValue ? int.MaxValue : Math.Abs(a);

    internal static int Square(int a) => unchecked(a * a);

    internal static int Min(int a, int b) => Math.Min(a, b);

    internal static int Max(int a, int b) => Math.Max(a, b);

    internal static int GreaterThan(int a, int b) => a > b ? 1 : 0;

    internal static int Select(int c, int a, int b) => c > 0 ? a : b;
}

/// <summary>Element-wise helpers for the array built-ins. Operands must share one length.</summary>
internal static class ElementWise
{
    internal static T[] Map1<T>(T[] a, Func<T, T> f)
    {
        var result = new T[a.Length];
        for (var i = 0; i < a.Length; i++)
        {
            result[i] = f(a[i]);
        }

        return result;
    }

    internal static T[] Map2<T>(T[] a, T[] b, Func<T, T, T> f)
    {
        RequireSameLength(a, b);
        var result = new T[a.Length];
        for (var i = 0; i < a.Length; i++)
        {
            result[i] = f(a[i], b[i]);
        }

        return result;
    }

    internal static T[] Map3<T>(T[] c, T[] a, T[] b, Func<T, T, T, T> f)
    {
        RequireSameLength(c, a);
        RequireSameLength(c, b);
        var result = new T[c.Length];
        for (var i = 0; i < c.Length; i++)
        {
            result[i] = f(c[i], a[i], b[i]);
        }

        return result;
    }

    private static void RequireSameLength<T>(T[] left, T[] right)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException(
                $"Array operands must have the same length; got {left.Length} and {right.Length}. No broadcasting is performed.");
        }
    }
}

/// <summary>Standard function set for <c>double</c> (Add, Subtract, Multiply, ProtectedDivide, Negate, Abs, Square, ProtectedSqrt, ProtectedLog, Min, Max, GreaterThan, Select, Constant).</summary>
public sealed class StandardDoubleFunctionSet : IFunctionSet<double>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly StandardDoubleFunctionSet Instance = new();

    private readonly StandardFunctionSet<double> _set = new("standard-double", new IFunction<double>[]
    {
        new StandardFunction<double>("Add", 2, (a, _) => DoubleOps.Add(a[0], a[1])),
        new StandardFunction<double>("Subtract", 2, (a, _) => DoubleOps.Subtract(a[0], a[1])),
        new StandardFunction<double>("Multiply", 2, (a, _) => DoubleOps.Multiply(a[0], a[1])),
        new StandardFunction<double>("ProtectedDivide", 2, (a, _) => DoubleOps.ProtectedDivide(a[0], a[1])),
        new StandardFunction<double>("Negate", 1, (a, _) => DoubleOps.Negate(a[0])),
        new StandardFunction<double>("Abs", 1, (a, _) => DoubleOps.Abs(a[0])),
        new StandardFunction<double>("Square", 1, (a, _) => DoubleOps.Square(a[0])),
        new StandardFunction<double>("ProtectedSqrt", 1, (a, _) => DoubleOps.ProtectedSqrt(a[0])),
        new StandardFunction<double>("ProtectedLog", 1, (a, _) => DoubleOps.ProtectedLog(a[0])),
        new StandardFunction<double>("Min", 2, (a, _) => DoubleOps.Min(a[0], a[1])),
        new StandardFunction<double>("Max", 2, (a, _) => DoubleOps.Max(a[0], a[1])),
        new StandardFunction<double>("GreaterThan", 2, (a, _) => DoubleOps.GreaterThan(a[0], a[1])),
        new StandardFunction<double>("Select", 3, (a, _) => DoubleOps.Select(a[0], a[1], a[2])),
        new StandardFunction<double>("Constant", 0, (_, p) => p),
    });

    /// <inheritdoc />
    public string Id => _set.Id;

    /// <inheritdoc />
    public int Count => _set.Count;

    /// <inheritdoc />
    public IFunction<double> Get(int functionId) => _set.Get(functionId);
}

/// <summary>Standard function set for <c>int</c> (no Sqrt/Log in v1).</summary>
public sealed class StandardInt32FunctionSet : IFunctionSet<int>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly StandardInt32FunctionSet Instance = new();

    private readonly StandardFunctionSet<int> _set = new("standard-int32", new IFunction<int>[]
    {
        new StandardFunction<int>("Add", 2, (a, _) => Int32Ops.Add(a[0], a[1])),
        new StandardFunction<int>("Subtract", 2, (a, _) => Int32Ops.Subtract(a[0], a[1])),
        new StandardFunction<int>("Multiply", 2, (a, _) => Int32Ops.Multiply(a[0], a[1])),
        new StandardFunction<int>("ProtectedDivide", 2, (a, _) => Int32Ops.ProtectedDivide(a[0], a[1])),
        new StandardFunction<int>("Negate", 1, (a, _) => Int32Ops.Negate(a[0])),
        new StandardFunction<int>("Abs", 1, (a, _) => Int32Ops.Abs(a[0])),
        new StandardFunction<int>("Square", 1, (a, _) => Int32Ops.Square(a[0])),
        new StandardFunction<int>("Min", 2, (a, _) => Int32Ops.Min(a[0], a[1])),
        new StandardFunction<int>("Max", 2, (a, _) => Int32Ops.Max(a[0], a[1])),
        new StandardFunction<int>("GreaterThan", 2, (a, _) => Int32Ops.GreaterThan(a[0], a[1])),
        new StandardFunction<int>("Select", 3, (a, _) => Int32Ops.Select(a[0], a[1], a[2])),
        new StandardFunction<int>("Constant", 0, (_, p) => p),
    });

    /// <inheritdoc />
    public string Id => _set.Id;

    /// <inheritdoc />
    public int Count => _set.Count;

    /// <inheritdoc />
    public IFunction<int> Get(int functionId) => _set.Get(functionId);
}

/// <summary>Element-wise standard function set for <c>double[]</c>. Constant returns a deep clone of the parameter.</summary>
public sealed class StandardDoubleArrayFunctionSet : IFunctionSet<double[]>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly StandardDoubleArrayFunctionSet Instance = new();

    private readonly StandardFunctionSet<double[]> _set = new("standard-double-array", new IFunction<double[]>[]
    {
        new StandardFunction<double[]>("Add", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.Add)),
        new StandardFunction<double[]>("Subtract", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.Subtract)),
        new StandardFunction<double[]>("Multiply", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.Multiply)),
        new StandardFunction<double[]>("ProtectedDivide", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.ProtectedDivide)),
        new StandardFunction<double[]>("Negate", 1, (a, _) => ElementWise.Map1(a[0], DoubleOps.Negate)),
        new StandardFunction<double[]>("Abs", 1, (a, _) => ElementWise.Map1(a[0], DoubleOps.Abs)),
        new StandardFunction<double[]>("Square", 1, (a, _) => ElementWise.Map1(a[0], DoubleOps.Square)),
        new StandardFunction<double[]>("ProtectedSqrt", 1, (a, _) => ElementWise.Map1(a[0], DoubleOps.ProtectedSqrt)),
        new StandardFunction<double[]>("ProtectedLog", 1, (a, _) => ElementWise.Map1(a[0], DoubleOps.ProtectedLog)),
        new StandardFunction<double[]>("Min", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.Min)),
        new StandardFunction<double[]>("Max", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.Max)),
        new StandardFunction<double[]>("GreaterThan", 2, (a, _) => ElementWise.Map2(a[0], a[1], DoubleOps.GreaterThan)),
        new StandardFunction<double[]>("Select", 3, (a, _) => ElementWise.Map3(a[0], a[1], a[2], DoubleOps.Select)),
        new StandardFunction<double[]>("Constant", 0, (_, p) => (double[])p.Clone()),
    });

    /// <inheritdoc />
    public string Id => _set.Id;

    /// <inheritdoc />
    public int Count => _set.Count;

    /// <inheritdoc />
    public IFunction<double[]> Get(int functionId) => _set.Get(functionId);
}

/// <summary>Element-wise standard function set for <c>int[]</c>. Constant returns a deep clone of the parameter.</summary>
public sealed class StandardInt32ArrayFunctionSet : IFunctionSet<int[]>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly StandardInt32ArrayFunctionSet Instance = new();

    private readonly StandardFunctionSet<int[]> _set = new("standard-int32-array", new IFunction<int[]>[]
    {
        new StandardFunction<int[]>("Add", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.Add)),
        new StandardFunction<int[]>("Subtract", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.Subtract)),
        new StandardFunction<int[]>("Multiply", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.Multiply)),
        new StandardFunction<int[]>("ProtectedDivide", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.ProtectedDivide)),
        new StandardFunction<int[]>("Negate", 1, (a, _) => ElementWise.Map1(a[0], Int32Ops.Negate)),
        new StandardFunction<int[]>("Abs", 1, (a, _) => ElementWise.Map1(a[0], Int32Ops.Abs)),
        new StandardFunction<int[]>("Square", 1, (a, _) => ElementWise.Map1(a[0], Int32Ops.Square)),
        new StandardFunction<int[]>("Min", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.Min)),
        new StandardFunction<int[]>("Max", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.Max)),
        new StandardFunction<int[]>("GreaterThan", 2, (a, _) => ElementWise.Map2(a[0], a[1], Int32Ops.GreaterThan)),
        new StandardFunction<int[]>("Select", 3, (a, _) => ElementWise.Map3(a[0], a[1], a[2], Int32Ops.Select)),
        new StandardFunction<int[]>("Constant", 0, (_, p) => (int[])p.Clone()),
    });

    /// <inheritdoc />
    public string Id => _set.Id;

    /// <inheritdoc />
    public int Count => _set.Count;

    /// <inheritdoc />
    public IFunction<int[]> Get(int functionId) => _set.Get(functionId);
}
