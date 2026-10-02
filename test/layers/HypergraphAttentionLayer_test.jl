using Test
using Lux
using Random

@testset "HypergraphAttentionLayer" begin

    @testset "Constructor" begin
        layer = HypergraphAttentionLayer(4, 8)

        @test layer.input_dim == 4
        @test layer.output_dim == 8
        @test layer.activation === tanh
        @test layer.use_bias
        @test layer.negative_slope == 0.2f0

        @test_throws ArgumentError HypergraphAttentionLayer(0, 8)
        @test_throws ArgumentError HypergraphAttentionLayer(-1, 8)
        @test_throws ArgumentError HypergraphAttentionLayer(4, 0)
        @test_throws ArgumentError HypergraphAttentionLayer(4, -1)

        @test_throws ArgumentError HypergraphAttentionLayer(
            4,
            8;
            negative_slope = -0.1f0,
        )
    end


    @testset "Constructor options" begin
        layer = HypergraphAttentionLayer(
            3,
            5;
            activation = identity,
            use_bias = false,
            negative_slope = 0.1f0,
        )

        @test layer.input_dim == 3
        @test layer.output_dim == 5
        @test layer.activation === identity
        @test !layer.use_bias
        @test layer.negative_slope == 0.1f0
    end


    @testset "Parameter and state initialization" begin
        layer = HypergraphAttentionLayer(4, 8)

        ps, st = Lux.setup(
            MersenneTwister(1),
            layer,
        )

        @test size(ps.W_vertex) == (4, 8)
        @test size(ps.W_output) == (8, 8)
        @test size(ps.a_vertex) == (8,)
        @test size(ps.a_hyperedge) == (8,)
        @test size(ps.b) == (1, 8)

        expected_parameter_count =
            4 * 8 +
            8 * 8 +
            8 +
            8 +
            8

        @test Lux.parameterlength(layer) ==
              expected_parameter_count

        @test Lux.parameterlength(ps) ==
              expected_parameter_count

        @test Lux.statelength(layer) == 0
        @test isempty(st)
    end


    @testset "Parameters without bias" begin
        layer = HypergraphAttentionLayer(
            4,
            8;
            use_bias = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(2),
            layer,
        )

        @test size(ps.W_vertex) == (4, 8)
        @test size(ps.W_output) == (8, 8)
        @test size(ps.a_vertex) == (8,)
        @test size(ps.a_hyperedge) == (8,)
        @test !hasproperty(ps, :b)

        expected_parameter_count =
            4 * 8 +
            8 * 8 +
            8 +
            8

        @test Lux.parameterlength(layer) ==
              expected_parameter_count

        @test Lux.parameterlength(ps) ==
              expected_parameter_count

        @test Lux.statelength(layer) == 0
        @test isempty(st)
    end


    @testset "Binary membership" begin
        H = Float32[
             1   0  -2;
             0   3   0;
             4   0   5
        ]

        membership =
            HyperGraphNeuralNetworks._hypergraph_attention_membership(
                H,
            )

        expected = Bool[
            1  0  1;
            0  1  0;
            1  0  1
        ]

        @test membership == expected
    end


    @testset "LeakyReLU attention function" begin
        f =
            HyperGraphNeuralNetworks._hypergraph_attention_leaky_relu

        @test f(2.0f0, 0.2f0) == 2.0f0
        @test f(0.0f0, 0.2f0) == 0.0f0
        @test f(-2.0f0, 0.2f0) ≈ -0.4f0
    end


    @testset "Masked softmax" begin
        scores = Float32[
            1,
            2,
            3,
            4,
        ]

        mask = Bool[
            true,
            false,
            true,
            false,
        ]

        result =
            HyperGraphNeuralNetworks._hypergraph_attention_masked_softmax(
                scores,
                mask,
            )

        @test length(result) == 4
        @test result[2] == 0
        @test result[4] == 0
        @test result[1] > 0
        @test result[3] > 0

        @test isapprox(
            sum(result),
            1.0f0;
            atol = 1f-6,
        )

        @test result[3] > result[1]
    end


    @testset "Masked softmax with empty mask" begin
        scores = Float32[
            1,
            2,
            3,
        ]

        mask = Bool[
            false,
            false,
            false,
        ]

        result =
            HyperGraphNeuralNetworks._hypergraph_attention_masked_softmax(
                scores,
                mask,
            )

        @test result == zeros(Float32, 3)
        @test all(isfinite, result)
    end


    @testset "Masked softmax numerical stability" begin
        scores = Float32[
            1000,
            1001,
            1002,
        ]

        mask = Bool[
            true,
            true,
            true,
        ]

        result =
            HyperGraphNeuralNetworks._hypergraph_attention_masked_softmax(
                scores,
                mask,
            )

        @test all(isfinite, result)

        @test isapprox(
            sum(result),
            1.0f0;
            atol = 1f-6,
        )

        @test result[3] > result[2]
        @test result[2] > result[1]
    end


    @testset "Masked softmax dimension validation" begin
        scores = Float32[
            1,
            2,
            3,
        ]

        mask = Bool[
            true,
            false,
        ]

        @test_throws DimensionMismatch begin
            HyperGraphNeuralNetworks._hypergraph_attention_masked_softmax(
                scores,
                mask,
            )
        end
    end


    @testset "Vertex-to-hyperedge attention" begin
        X = Float32[
            1  0;
            0  1;
            1  1
        ]

        H = Float32[
            1  0;
            1  1;
            0  1
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        hyperedge_messages,
        attention =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H,
                attention_vector,
                0.2f0,
            )

        @test size(hyperedge_messages) == (2, 2)
        @test size(attention) == (3, 2)

        @test attention[3, 1] == 0
        @test attention[1, 2] == 0

        @test isapprox(
            sum(attention[:, 1]),
            1.0f0;
            atol = 1f-6,
        )

        @test isapprox(
            sum(attention[:, 2]),
            1.0f0;
            atol = 1f-6,
        )

        @test all(isfinite, hyperedge_messages)
        @test all(isfinite, attention)
    end


    @testset "Vertex-to-hyperedge attention with equal scores" begin
        X = Float32[
            1  0;
            1  0;
            5  0
        ]

        # IMPORTANT:
        # This must be a 3 × 1 incidence MATRIX, not a Vector.
        H = reshape(
            Float32[
                1,
                1,
                0,
            ],
            3,
            1,
        )

        attention_vector = Float32[
            1,
            0,
        ]

        hyperedge_messages,
        attention =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H,
                attention_vector,
                0.2f0,
            )

        @test size(H) == (3, 1)
        @test size(attention) == (3, 1)
        @test size(hyperedge_messages) == (1, 2)

        @test isapprox(
            attention[1, 1],
            0.5f0;
            atol = 1f-6,
        )

        @test isapprox(
            attention[2, 1],
            0.5f0;
            atol = 1f-6,
        )

        @test attention[3, 1] == 0

        @test isapprox(
            hyperedge_messages[1, :],
            Float32[1, 0];
            atol = 1f-6,
        )
    end


    @testset "Vertex-to-hyperedge empty hyperedge" begin
        X = Float32[
            1  0;
            0  1;
            1  1
        ]

        H = Float32[
            1  0;
            1  0;
            0  0
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        hyperedge_messages,
        attention =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H,
                attention_vector,
                0.2f0,
            )

        @test attention[:, 2] == zeros(Float32, 3)
        @test hyperedge_messages[2, :] == zeros(Float32, 2)

        @test all(isfinite, hyperedge_messages)
        @test all(isfinite, attention)
    end


    @testset "Vertex-to-hyperedge attention vector validation" begin
        X = Float32[
            1  0;
            0  1
        ]

        # IMPORTANT:
        # This must be a 2 × 1 incidence MATRIX, not a Vector.
        H = reshape(
            Float32[
                1,
                1,
            ],
            2,
            1,
        )

        bad_attention_vector = Float32[
            1,
            2,
            3,
        ]

        @test size(H) == (2, 1)

        @test_throws DimensionMismatch begin
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H,
                bad_attention_vector,
                0.2f0,
            )
        end
    end


    @testset "Hyperedge-to-vertex attention" begin
        hyperedge_messages = Float32[
            1  0;
            0  1
        ]

        H = Float32[
            1  0;
            1  1;
            0  1
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        propagated_vertices,
        attention =
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                attention_vector,
                0.2f0,
            )

        @test size(propagated_vertices) == (3, 2)
        @test size(attention) == (3, 2)

        @test attention[1, 2] == 0
        @test attention[3, 1] == 0

        @test isapprox(
            sum(attention[1, :]),
            1.0f0;
            atol = 1f-6,
        )

        @test isapprox(
            sum(attention[2, :]),
            1.0f0;
            atol = 1f-6,
        )

        @test isapprox(
            sum(attention[3, :]),
            1.0f0;
            atol = 1f-6,
        )

        @test all(isfinite, propagated_vertices)
        @test all(isfinite, attention)
    end


    @testset "Hyperedge-to-vertex equal attention" begin
        hyperedge_messages = Float32[
            1  0;
            0  1
        ]

        H = Float32[
            1  1
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        propagated_vertices,
        attention =
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                attention_vector,
                0.2f0,
            )

        @test isapprox(
            attention[1, 1],
            0.5f0;
            atol = 1f-6,
        )

        @test isapprox(
            attention[1, 2],
            0.5f0;
            atol = 1f-6,
        )

        @test isapprox(
            propagated_vertices[1, :],
            Float32[0.5, 0.5];
            atol = 1f-6,
        )
    end


    @testset "Isolated vertex" begin
        hyperedge_messages = Float32[
            1  0;
            0  1
        ]

        H = Float32[
            1  0;
            0  0;
            0  1
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        propagated_vertices,
        attention =
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                attention_vector,
                0.2f0,
            )

        @test attention[2, :] == zeros(Float32, 2)
        @test propagated_vertices[2, :] == zeros(Float32, 2)

        @test all(isfinite, propagated_vertices)
        @test all(isfinite, attention)
    end


    @testset "Hyperedge count validation" begin
        hyperedge_messages = Float32[
            1  0;
            0  1;
            1  1
        ]

        H = Float32[
            1  0;
            1  1
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        @test_throws DimensionMismatch begin
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                attention_vector,
                0.2f0,
            )
        end
    end


    @testset "Hyperedge attention vector validation" begin
        hyperedge_messages = Float32[
            1  0;
            0  1
        ]

        H = Float32[
            1  0;
            1  1
        ]

        bad_attention_vector = Float32[
            1,
            2,
            3,
        ]

        @test_throws DimensionMismatch begin
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                bad_attention_vector,
                0.2f0,
            )
        end
    end


    @testset "Non-binary incidence represents membership" begin
        X = Float32[
            1  0;
            0  1;
            1  1
        ]

        H_binary = Float32[
            1  0;
            1  1;
            0  1
        ]

        H_nonbinary = Float32[
             2   0;
            -4   7;
             0  -3
        ]

        attention_vector = Float32[
            1,
            1,
        ]

        messages_binary,
        attention_binary =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H_binary,
                attention_vector,
                0.2f0,
            )

        messages_nonbinary,
        attention_nonbinary =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                X,
                H_nonbinary,
                attention_vector,
                0.2f0,
            )

        @test isapprox(
            messages_binary,
            messages_nonbinary,
        )

        @test isapprox(
            attention_binary,
            attention_nonbinary,
        )
    end


    @testset "Forward pass" begin
        layer = HypergraphAttentionLayer(
            4,
            8,
        )

        ps, st = Lux.setup(
            MersenneTwister(3),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(4),
                Float32,
                5,
                4,
            )

        H = Float32[
            1  0  0;
            1  1  0;
            0  1  1;
            0  0  1;
            1  0  1
        ]

        output, st_new =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        @test size(output.updated_vertices) == (5, 8)
        @test size(output.hyperedge_messages) == (3, 8)

        @test size(
            output.vertex_to_hyperedge_attention,
        ) == (5, 3)

        @test size(
            output.hyperedge_to_vertex_attention,
        ) == (5, 3)

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.hyperedge_messages)

        @test all(
            isfinite,
            output.vertex_to_hyperedge_attention,
        )

        @test all(
            isfinite,
            output.hyperedge_to_vertex_attention,
        )

        @test st_new == st
    end


    @testset "Attention respects incidence structure" begin
        layer = HypergraphAttentionLayer(
            3,
            5,
        )

        ps, st = Lux.setup(
            MersenneTwister(5),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(6),
                Float32,
                4,
                3,
            )

        H = Float32[
            1  0;
            1  1;
            0  1;
            0  0
        ]

        output, _ =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        membership =
            H .!= 0

        @test all(
            output.vertex_to_hyperedge_attention[
                .!membership
            ] .== 0,
        )

        @test all(
            output.hyperedge_to_vertex_attention[
                .!membership
            ] .== 0,
        )
    end


    @testset "Vertex-to-hyperedge attention normalization" begin
        layer = HypergraphAttentionLayer(
            3,
            5,
        )

        ps, st = Lux.setup(
            MersenneTwister(7),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(8),
                Float32,
                4,
                3,
            )

        H = Float32[
            1  0;
            1  1;
            0  1;
            1  0
        ]

        output, _ =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        attention =
            output.vertex_to_hyperedge_attention

        for hyperedge_index in axes(H, 2)
            if any(
                H[:, hyperedge_index] .!= 0
            )
                @test isapprox(
                    sum(
                        attention[
                            :,
                            hyperedge_index,
                        ],
                    ),
                    1.0f0;
                    atol = 1f-5,
                )
            end
        end
    end


    @testset "Hyperedge-to-vertex attention normalization" begin
        layer = HypergraphAttentionLayer(
            3,
            5,
        )

        ps, st = Lux.setup(
            MersenneTwister(9),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(10),
                Float32,
                4,
                3,
            )

        H = Float32[
            1  0;
            1  1;
            0  1;
            1  0
        ]

        output, _ =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        attention =
            output.hyperedge_to_vertex_attention

        for vertex_index in axes(H, 1)
            if any(
                H[vertex_index, :] .!= 0
            )
                @test isapprox(
                    sum(
                        attention[
                            vertex_index,
                            :,
                        ],
                    ),
                    1.0f0;
                    atol = 1f-5,
                )
            end
        end
    end


    @testset "Forward pass without bias" begin
        layer = HypergraphAttentionLayer(
            3,
            5;
            use_bias = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(11),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(12),
                Float32,
                4,
                3,
            )

        H = Float32[
            1  0;
            1  1;
            0  1;
            1  0
        ]

        output, st_new =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        @test size(output.updated_vertices) == (4, 5)
        @test size(output.hyperedge_messages) == (2, 5)

        @test !hasproperty(ps, :b)
        @test all(isfinite, output.updated_vertices)
        @test st_new == st
    end


    @testset "Identity activation" begin
        layer = HypergraphAttentionLayer(
            2,
            2;
            activation = identity,
            use_bias = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(13),
            layer,
        )

        X_vertex = Float32[
            1  2;
            3  4;
            5  6
        ]

        H = Float32[
            1  0;
            1  1;
            0  1
        ]

        output, _ =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        transformed_vertices =
            X_vertex *
            ps.W_vertex

        hyperedge_messages,
        _ =
            HyperGraphNeuralNetworks._hypergraph_vertex_to_hyperedge_attention(
                transformed_vertices,
                H,
                ps.a_vertex,
                layer.negative_slope,
            )

        propagated_vertices,
        _ =
            HyperGraphNeuralNetworks._hypergraph_hyperedge_to_vertex_attention(
                hyperedge_messages,
                H,
                ps.a_hyperedge,
                layer.negative_slope,
            )

        expected =
            propagated_vertices *
            ps.W_output

        @test isapprox(
            output.updated_vertices,
            expected;
            atol = 1f-6,
        )
    end


    @testset "Isolated vertices and empty hyperedges" begin
        layer = HypergraphAttentionLayer(
            3,
            4,
        )

        ps, st = Lux.setup(
            MersenneTwister(14),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(15),
                Float32,
                4,
                3,
            )

        H = Float32[
            1  0  0;
            0  0  0;
            1  0  0;
            0  0  0
        ]

        output, _ =
            layer(
                (
                    X_vertex,
                    H,
                ),
                ps,
                st,
            )

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.hyperedge_messages)

        @test all(
            isfinite,
            output.vertex_to_hyperedge_attention,
        )

        @test all(
            isfinite,
            output.hyperedge_to_vertex_attention,
        )

        @test output.hyperedge_messages[2, :] ==
              zeros(Float32, 4)

        @test output.hyperedge_messages[3, :] ==
              zeros(Float32, 4)

        @test output.vertex_to_hyperedge_attention[:, 2] ==
              zeros(Float32, 4)

        @test output.vertex_to_hyperedge_attention[:, 3] ==
              zeros(Float32, 4)

        @test output.hyperedge_to_vertex_attention[2, :] ==
              zeros(Float32, 3)

        @test output.hyperedge_to_vertex_attention[4, :] ==
              zeros(Float32, 3)
    end


    @testset "Input validation" begin
        layer = HypergraphAttentionLayer(
            4,
            8,
        )

        ps, st = Lux.setup(
            MersenneTwister(16),
            layer,
        )

        X_vertex =
            rand(
                Float32,
                5,
                4,
            )

        H =
            ones(
                Float32,
                5,
                3,
            )

        @test_throws ArgumentError layer(
            (X_vertex,),
            ps,
            st,
        )

        @test_throws ArgumentError layer(
            (
                X_vertex,
                H,
                H,
            ),
            ps,
            st,
        )

        bad_vertex_features =
            rand(
                Float32,
                5,
                5,
            )

        @test_throws DimensionMismatch layer(
            (
                bad_vertex_features,
                H,
            ),
            ps,
            st,
        )

        bad_incidence =
            ones(
                Float32,
                4,
                3,
            )

        @test_throws DimensionMismatch layer(
            (
                X_vertex,
                bad_incidence,
            ),
            ps,
            st,
        )
    end


    @testset "Deterministic initialization" begin
        layer = HypergraphAttentionLayer(
            4,
            8,
        )

        ps_1, _ =
            Lux.setup(
                MersenneTwister(42),
                layer,
            )

        ps_2, _ =
            Lux.setup(
                MersenneTwister(42),
                layer,
            )

        @test ps_1.W_vertex == ps_2.W_vertex
        @test ps_1.W_output == ps_2.W_output
        @test ps_1.a_vertex == ps_2.a_vertex
        @test ps_1.a_hyperedge == ps_2.a_hyperedge
        @test ps_1.b == ps_2.b
    end


    @testset "Deterministic initialization without bias" begin
        layer = HypergraphAttentionLayer(
            4,
            8;
            use_bias = false,
        )

        ps_1, _ =
            Lux.setup(
                MersenneTwister(42),
                layer,
            )

        ps_2, _ =
            Lux.setup(
                MersenneTwister(42),
                layer,
            )

        @test ps_1.W_vertex == ps_2.W_vertex
        @test ps_1.W_output == ps_2.W_output
        @test ps_1.a_vertex == ps_2.a_vertex
        @test ps_1.a_hyperedge == ps_2.a_hyperedge

        @test !hasproperty(
            ps_1,
            :b,
        )

        @test !hasproperty(
            ps_2,
            :b,
        )
    end
end