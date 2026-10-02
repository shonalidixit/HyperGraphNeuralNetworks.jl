using Test
using Lux
using Random

@testset "UniGNNHypergraphLayer" begin

    @testset "Constructor" begin
        layer = UniGNNHypergraphLayer(4, 8)

        @test layer.input_dim == 4
        @test layer.output_dim == 8
        @test layer.activation === tanh
        @test layer.use_bias
        @test layer.include_self

        @test_throws ArgumentError UniGNNHypergraphLayer(0, 8)
        @test_throws ArgumentError UniGNNHypergraphLayer(-1, 8)
        @test_throws ArgumentError UniGNNHypergraphLayer(4, 0)
        @test_throws ArgumentError UniGNNHypergraphLayer(4, -1)
    end

    @testset "Constructor options" begin
        layer = UniGNNHypergraphLayer(
            3,
            5;
            activation = identity,
            use_bias = false,
            include_self = false,
        )

        @test layer.input_dim == 3
        @test layer.output_dim == 5
        @test layer.activation === identity
        @test !layer.use_bias
        @test !layer.include_self
    end

    @testset "Parameter and state initialization" begin
        layer = UniGNNHypergraphLayer(4, 8)
        ps, st = Lux.setup(
            MersenneTwister(1),
            layer,
        )

        @test size(ps.W) == (8, 8)
        @test size(ps.b) == (1, 8)

        expected_parameter_count =
            8 * 8 + 8

        @test Lux.parameterlength(layer) ==
              expected_parameter_count

        @test Lux.parameterlength(ps) ==
              expected_parameter_count

        @test Lux.statelength(layer) == 0
        @test isempty(st)
    end

    @testset "Parameters without self features" begin
        layer = UniGNNHypergraphLayer(
            4,
            8;
            include_self = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(2),
            layer,
        )

        @test size(ps.W) == (4, 8)
        @test size(ps.b) == (1, 8)

        expected_parameter_count =
            4 * 8 + 8

        @test Lux.parameterlength(layer) ==
              expected_parameter_count

        @test Lux.parameterlength(ps) ==
              expected_parameter_count

        @test Lux.statelength(layer) == 0
        @test isempty(st)
    end

    @testset "Parameters without bias" begin
        layer = UniGNNHypergraphLayer(
            4,
            8;
            use_bias = false,
        )

        ps, _ = Lux.setup(
            MersenneTwister(3),
            layer,
        )

        @test size(ps.W) == (8, 8)
        @test !hasproperty(ps, :b)

        expected_parameter_count =
            8 * 8

        @test Lux.parameterlength(layer) ==
              expected_parameter_count

        @test Lux.parameterlength(ps) ==
              expected_parameter_count
    end

    @testset "Membership matrix" begin
        H = Float32[
            1  0  -2;
            0  3   0;
            4  0   5
        ]

        expected = Float32[
            1  0  1;
            0  1  0;
            1  0  1
        ]

        result =
            HyperGraphNeuralNetworks._unignn_membership_matrix(
                H,
            )

        @test result == expected
    end

    @testset "Safe inverse" begin
        values =
            Float32[
                1,
                2,
                0,
                4,
            ]

        result =
            HyperGraphNeuralNetworks._unignn_safe_inverse(
                values,
            )

        expected =
            Float32[
                1,
                0.5,
                0,
                0.25,
            ]

        @test isapprox(
            result,
            expected,
        )

        @test all(
            isfinite,
            result,
        )
    end

    @testset "Vertex-to-hyperedge aggregation" begin
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

        result =
            HyperGraphNeuralNetworks._unignn_vertex_to_hyperedge(
                X_vertex,
                H,
            )

        expected = Float32[
            2  3;
            4  5
        ]

        @test size(result) == (2, 2)

        @test isapprox(
            result,
            expected,
        )
    end

    @testset "Hyperedge-to-vertex aggregation" begin
        hyperedge_messages = Float32[
            2  3;
            4  5
        ]

        H = Float32[
            1  0;
            1  1;
            0  1
        ]

        result =
            HyperGraphNeuralNetworks._unignn_hyperedge_to_vertex(
                hyperedge_messages,
                H,
            )

        expected = Float32[
            2  3;
            3  4;
            4  5
        ]

        @test size(result) == (3, 2)

        @test isapprox(
            result,
            expected,
        )
    end

    @testset "Two-stage aggregation" begin
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

        hyperedge_messages =
            HyperGraphNeuralNetworks._unignn_vertex_to_hyperedge(
                X_vertex,
                H,
            )

        propagated_vertices =
            HyperGraphNeuralNetworks._unignn_hyperedge_to_vertex(
                hyperedge_messages,
                H,
            )

        expected_hyperedges = Float32[
            2  3;
            4  5
        ]

        expected_vertices = Float32[
            2  3;
            3  4;
            4  5
        ]

        @test isapprox(
            hyperedge_messages,
            expected_hyperedges,
        )

        @test isapprox(
            propagated_vertices,
            expected_vertices,
        )
    end

    @testset "Non-binary incidence values represent membership" begin
        X_vertex = Float32[
            1  2;
            3  4;
            5  6
        ]

        H = Float32[
             2   0;
            -3   7;
             0  -1
        ]

        hyperedge_messages =
            HyperGraphNeuralNetworks._unignn_vertex_to_hyperedge(
                X_vertex,
                H,
            )

        propagated_vertices =
            HyperGraphNeuralNetworks._unignn_hyperedge_to_vertex(
                hyperedge_messages,
                H,
            )

        expected_hyperedges = Float32[
            2  3;
            4  5
        ]

        expected_vertices = Float32[
            2  3;
            3  4;
            4  5
        ]

        @test isapprox(
            hyperedge_messages,
            expected_hyperedges,
        )

        @test isapprox(
            propagated_vertices,
            expected_vertices,
        )
    end

    @testset "Zero-degree vertices and empty hyperedges" begin
        X_vertex = Float32[
            1  2;
            3  4;
            5  6;
            7  8
        ]

        H = Float32[
            1  0  0;
            0  0  0;
            1  0  0;
            0  0  0
        ]

        hyperedge_messages =
            HyperGraphNeuralNetworks._unignn_vertex_to_hyperedge(
                X_vertex,
                H,
            )

        propagated_vertices =
            HyperGraphNeuralNetworks._unignn_hyperedge_to_vertex(
                hyperedge_messages,
                H,
            )

        expected_hyperedges = Float32[
            3  4;
            0  0;
            0  0
        ]

        expected_vertices = Float32[
            3  4;
            0  0;
            3  4;
            0  0
        ]

        @test all(
            isfinite,
            hyperedge_messages,
        )

        @test all(
            isfinite,
            propagated_vertices,
        )

        @test isapprox(
            hyperedge_messages,
            expected_hyperedges,
        )

        @test isapprox(
            propagated_vertices,
            expected_vertices,
        )
    end

    @testset "Forward pass" begin
        layer = UniGNNHypergraphLayer(
            4,
            8;
            activation = tanh,
            use_bias = true,
            include_self = true,
        )

        ps, st = Lux.setup(
            MersenneTwister(4),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(5),
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
                (X_vertex, H),
                ps,
                st,
            )

        @test size(
            output.updated_vertices,
        ) == (5, 8)

        @test size(
            output.hyperedge_messages,
        ) == (3, 4)

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.hyperedge_messages,
        )

        @test st_new == st
    end

    @testset "Forward pass without self features" begin
        layer = UniGNNHypergraphLayer(
            4,
            6;
            include_self = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(6),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(7),
                Float32,
                4,
                4,
            )

        H = Float32[
            1  0;
            1  1;
            0  1;
            1  0
        ]

        output, st_new =
            layer(
                (X_vertex, H),
                ps,
                st,
            )

        @test size(
            output.updated_vertices,
        ) == (4, 6)

        @test size(
            output.hyperedge_messages,
        ) == (2, 4)

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.hyperedge_messages,
        )

        @test st_new == st
    end

    @testset "Forward pass without bias" begin
        layer = UniGNNHypergraphLayer(
            3,
            5;
            use_bias = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(8),
            layer,
        )

        X_vertex =
            rand(
                MersenneTwister(9),
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
                (X_vertex, H),
                ps,
                st,
            )

        @test size(
            output.updated_vertices,
        ) == (4, 5)

        @test size(
            output.hyperedge_messages,
        ) == (2, 3)

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.hyperedge_messages,
        )
    end

    @testset "Identity activation" begin
        layer = UniGNNHypergraphLayer(
            2,
            2;
            activation = identity,
            use_bias = false,
            include_self = false,
        )

        ps, st = Lux.setup(
            MersenneTwister(10),
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
                (X_vertex, H),
                ps,
                st,
            )

        hyperedge_messages =
            HyperGraphNeuralNetworks._unignn_vertex_to_hyperedge(
                X_vertex,
                H,
            )

        propagated_vertices =
            HyperGraphNeuralNetworks._unignn_hyperedge_to_vertex(
                hyperedge_messages,
                H,
            )

        expected =
            propagated_vertices *
            ps.W

        @test isapprox(
            output.updated_vertices,
            expected,
        )
    end

    @testset "Forward pass with isolated structure" begin
        layer =
            UniGNNHypergraphLayer(
                3,
                5,
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
            1  0  0;
            0  0  0;
            1  0  0;
            0  0  0
        ]

        output, _ =
            layer(
                (X_vertex, H),
                ps,
                st,
            )

        @test size(
            output.updated_vertices,
        ) == (4, 5)

        @test size(
            output.hyperedge_messages,
        ) == (3, 3)

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.hyperedge_messages,
        )

        @test isapprox(
            output.hyperedge_messages[2, :],
            zeros(Float32, 3),
        )

        @test isapprox(
            output.hyperedge_messages[3, :],
            zeros(Float32, 3),
        )
    end

    @testset "Input validation" begin
        layer =
            UniGNNHypergraphLayer(
                4,
                8,
            )

        ps, st = Lux.setup(
            MersenneTwister(13),
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
        layer =
            UniGNNHypergraphLayer(
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

        @test ps_1.W == ps_2.W
        @test ps_1.b == ps_2.b
    end

    @testset "Deterministic initialization without bias" begin
        layer =
            UniGNNHypergraphLayer(
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

        @test ps_1.W == ps_2.W
        @test !hasproperty(ps_1, :b)
        @test !hasproperty(ps_2, :b)
    end
end