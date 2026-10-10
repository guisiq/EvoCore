using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Reparo-r12 (achado 2 de review-r11, conclusão de R10-2): padrão oficial de "dynamic skip" do xunit v2
/// (amostra `DynamicSkipExample` do próprio projeto xunit), usado para que um teste dependente de hardware
/// (dispositivo OpenCL real) produza um resultado efetivo de "ignorado"/indisponível quando o runtime não
/// enumera nenhum <c>CLDevice</c>, em vez de um retorno normal de um <c>[Fact]</c> comum, que o runner contaria
/// como aprovado. Quando o dispositivo existe, o teste roda e falha/passa normalmente, sem nenhum fallback.
/// </summary>
public sealed class OptionalFactDiscoverer : IXunitTestCaseDiscoverer
{
    private readonly IMessageSink _diagnosticMessageSink;

    public OptionalFactDiscoverer(IMessageSink diagnosticMessageSink)
    {
        _diagnosticMessageSink = diagnosticMessageSink;
    }

    public IEnumerable<IXunitTestCase> Discover(ITestFrameworkDiscoveryOptions discoveryOptions, ITestMethod testMethod, IAttributeInfo factAttribute)
    {
        yield return new OptionalFactTestCase(_diagnosticMessageSink, discoveryOptions.MethodDisplayOrDefault(), discoveryOptions.MethodDisplayOptionsOrDefault(), testMethod);
    }
}

public sealed class OptionalFactTestCase : XunitTestCase
{
    [Obsolete("Chamado apenas pelo desserializador do xunit", error: true)]
    public OptionalFactTestCase()
    {
    }

    public OptionalFactTestCase(IMessageSink diagnosticMessageSink, TestMethodDisplay defaultMethodDisplay, TestMethodDisplayOptions defaultMethodDisplayOptions, ITestMethod testMethod, object[]? testMethodArguments = null)
        : base(diagnosticMessageSink, defaultMethodDisplay, defaultMethodDisplayOptions, testMethod, testMethodArguments)
    {
    }

    public override async Task<RunSummary> RunAsync(IMessageSink diagnosticMessageSink, IMessageBus messageBus, object[] constructorArguments, ExceptionAggregator aggregator, CancellationTokenSource cancellationTokenSource)
    {
        var skipMessageBus = new SkipMessageBus(messageBus);
        var result = await base.RunAsync(diagnosticMessageSink, skipMessageBus, constructorArguments, aggregator, cancellationTokenSource).ConfigureAwait(false);
        if (skipMessageBus.DynamicallySkippedCount > 0)
        {
            result.Failed -= skipMessageBus.DynamicallySkippedCount;
        }

        result.Skipped += skipMessageBus.DynamicallySkippedCount;
        return result;
    }
}

internal sealed class SkipMessageBus : IMessageBus
{
    private readonly IMessageBus _innerBus;

    public SkipMessageBus(IMessageBus innerBus)
    {
        _innerBus = innerBus;
    }

    public int DynamicallySkippedCount { get; private set; }

    public void Dispose()
    {
    }

    public bool QueueMessage(IMessageSinkMessage message)
    {
        if (message is ITestFailed testFailed)
        {
            var exceptionType = testFailed.ExceptionTypes.FirstOrDefault();
            if (exceptionType == typeof(DynamicSkipException).FullName)
            {
                DynamicallySkippedCount++;
                return _innerBus.QueueMessage(new TestSkipped(testFailed.Test, testFailed.Messages.FirstOrDefault() ?? "indisponível"));
            }
        }

        return _innerBus.QueueMessage(message);
    }
}

/// <summary>Sinalizada por <see cref="DynamicSkip.If"/> para converter um resultado em "ignorado" dinamicamente.</summary>
public sealed class DynamicSkipException : Exception
{
    public DynamicSkipException(string reason) : base(reason)
    {
    }
}

/// <summary>
/// <see cref="FactAttribute"/> cujo caso de teste pode ser convertido em "ignorado" em tempo de execução via
/// <see cref="DynamicSkip.If"/>, em vez de sempre contar como aprovado/reprovado.
/// </summary>
[XunitTestCaseDiscoverer("Evo.Ilgpu.Tests.OptionalFactDiscoverer", "Evo.Ilgpu.Tests")]
public sealed class OptionalFactAttribute : FactAttribute
{
}

public static class DynamicSkip
{
    /// <summary>Converte a execução atual em resultado "ignorado" (não aprovado, não reprovado) quando <paramref name="condition"/> é verdadeira.</summary>
    public static void If(bool condition, string reason)
    {
        if (condition)
        {
            throw new DynamicSkipException(reason);
        }
    }
}
