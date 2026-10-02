using Test

using Lux

using Random



@testset "SpatialHypergraphLayer" begin



    @testset "Constructor" begin

        layer = SpatialHypergraphLayer(4, 0, 8)



        @test layer.vertex_in_dim == 4

        @test layer.hyperedge_in_dim == 0

        @test layer.hidden_dim == 8

        @test layer.normalize

        @test layer.activation === tanh



        @test_throws ArgumentError SpatialHypergraphLayer(0, 0, 8)

        @test_throws ArgumentError SpatialHypergraphLayer(-1, 0, 8)

        @test_throws ArgumentError SpatialHypergraphLayer(4, -1, 8)

        @test_throws ArgumentError SpatialHypergraphLayer(4, 0, 0)

        @test_throws ArgumentError SpatialHypergraphLayer(4, 0, -1)

    end



    @testset "Parameter and state initialization" begin

        layer = SpatialHypergraphLayer(4, 0, 8)

        ps, st = Lux.setup(MersenneTwister(1), layer)



        @test size(ps.W_vertex) == (4, 8)

        @test size(ps.b_vertex) == (1, 8)

        @test size(ps.W_hyperedge) == (8, 8)

        @test size(ps.b_hyperedge) == (1, 8)

        @test size(ps.W_vertex_update) == (12, 8)

        @test size(ps.b_vertex_update) == (1, 8)



        expected_parameter_count =

            4 * 8 + 8 +

            8 * 8 + 8 +

            12 * 8 + 8



        @test Lux.parameterlength(layer) == expected_parameter_count

        @test Lux.parameterlength(ps) == expected_parameter_count

        @test Lux.statelength(layer) == 0

        @test isempty(st)

    end



    @testset "Parameter initialization with hyperedge features" begin

        layer = SpatialHypergraphLayer(4, 3, 8)

        ps, _ = Lux.setup(MersenneTwister(2), layer)



        @test size(ps.W_vertex) == (4, 8)

        @test size(ps.b_vertex) == (1, 8)

        @test size(ps.W_hyperedge) == (11, 8)

        @test size(ps.b_hyperedge) == (1, 8)

        @test size(ps.W_vertex_update) == (12, 8)

        @test size(ps.b_vertex_update) == (1, 8)



        expected_parameter_count =

            4 * 8 + 8 +

            11 * 8 + 8 +

            12 * 8 + 8



        @test Lux.parameterlength(layer) == expected_parameter_count

        @test Lux.parameterlength(ps) == expected_parameter_count

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



        @test HyperGraphNeuralNetworks._spatial_membership_matrix(H) == expected

    end



    @testset "Safe inverse" begin

        values = Float32[1, 2, 0, 4]

        result = HyperGraphNeuralNetworks._spatial_safe_inverse(values)



        expected = Float32[1, 0.5, 0, 0.25]



        @test isapprox(result, expected)

        @test all(isfinite, result)

    end



    @testset "Vertex-to-hyperedge propagation" begin

        H = Float32[

            1  0;

            1  1;

            0  1

        ]



        unnormalized =

            HyperGraphNeuralNetworks._spatial_vertex_to_hyperedge(H, false)



        expected_unnormalized = Float32[

            1  1  0;

            0  1  1

        ]



        @test unnormalized == expected_unnormalized



        normalized =

            HyperGraphNeuralNetworks._spatial_vertex_to_hyperedge(H, true)



        expected_normalized = Float32[

            0.5  0.5  0.0;

            0.0  0.5  0.5

        ]



        @test isapprox(normalized, expected_normalized)

    end



    @testset "Hyperedge-to-vertex propagation" begin

        H = Float32[

            1  0;

            1  1;

            0  1

        ]



        unnormalized =

            HyperGraphNeuralNetworks._spatial_hyperedge_to_vertex(H, false)



        @test unnormalized == H



        normalized =

            HyperGraphNeuralNetworks._spatial_hyperedge_to_vertex(H, true)



        expected_normalized = Float32[

            1.0  0.0;

            0.5  0.5;

            0.0  1.0

        ]



        @test isapprox(normalized, expected_normalized)

    end



    @testset "Non-binary incidence values represent membership" begin

        H = Float32[

             2   0;

            -3   7;

             0  -1

        ]



        normalized_v2e =

            HyperGraphNeuralNetworks._spatial_vertex_to_hyperedge(H, true)



        normalized_e2v =

            HyperGraphNeuralNetworks._spatial_hyperedge_to_vertex(H, true)



        expected_v2e = Float32[

            0.5  0.5  0.0;

            0.0  0.5  0.5

        ]



        expected_e2v = Float32[

            1.0  0.0;

            0.5  0.5;

            0.0  1.0

        ]



        @test isapprox(normalized_v2e, expected_v2e)

        @test isapprox(normalized_e2v, expected_e2v)

    end



    @testset "Zero-degree vertices and hyperedges" begin

        H = Float32[

            1  0  0;

            0  0  0;

            1  0  0

        ]



        v2e = HyperGraphNeuralNetworks._spatial_vertex_to_hyperedge(H, true)

        e2v = HyperGraphNeuralNetworks._spatial_hyperedge_to_vertex(H, true)



        @test all(isfinite, v2e)

        @test all(isfinite, e2v)



        expected_v2e = Float32[

            0.5  0.0  0.5;

            0.0  0.0  0.0;

            0.0  0.0  0.0

        ]



        expected_e2v = Float32[

            1.0  0.0  0.0;

            0.0  0.0  0.0;

            1.0  0.0  0.0

        ]



        @test isapprox(v2e, expected_v2e)

        @test isapprox(e2v, expected_e2v)

    end



    @testset "Forward pass without hyperedge features" begin

        layer = SpatialHypergraphLayer(

            4,

            0,

            8;

            activation = tanh,

            normalize = true,

        )



        ps, st = Lux.setup(MersenneTwister(3), layer)



        X_vertex =

            rand(MersenneTwister(4), Float32, 5, 4)



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



        @test size(output.updated_vertices) == (5, 8)

        @test size(output.updated_hyperedges) == (3, 8)

        @test all(isfinite, output.updated_vertices)

        @test all(isfinite, output.updated_hyperedges)

        @test st_new == st

    end



    @testset "Forward pass with hyperedge features" begin

        layer = SpatialHypergraphLayer(

            4,

            3,

            8;

            activation = tanh,

            normalize = true,

        )



        ps, st = Lux.setup(MersenneTwister(5), layer)



        X_vertex =

            rand(MersenneTwister(6), Float32, 5, 4)



        X_hyperedge =

            rand(MersenneTwister(7), Float32, 3, 3)



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

                    X_hyperedge,

                    H,

                ),

                ps,

                st,

            )



        @test size(output.updated_vertices) == (5, 8)

        @test size(output.updated_hyperedges) == (3, 8)

        @test all(isfinite, output.updated_vertices)

        @test all(isfinite, output.updated_hyperedges)

        @test st_new == st

    end



    @testset "Unnormalized forward pass" begin

        layer = SpatialHypergraphLayer(

            4,

            0,

            6;

            normalize = false,

        )



        ps, st = Lux.setup(MersenneTwister(8), layer)



        X_vertex =

            rand(MersenneTwister(9), Float32, 4, 4)



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



        @test size(output.updated_vertices) == (4, 6)

        @test size(output.updated_hyperedges) == (2, 6)

        @test all(isfinite, output.updated_vertices)

        @test all(isfinite, output.updated_hyperedges)

    end



    @testset "Forward pass with isolated structure" begin

        layer = SpatialHypergraphLayer(

            3,

            0,

            5;

            normalize = true,

        )



        ps, st = Lux.setup(MersenneTwister(10), layer)



        X_vertex =

            rand(MersenneTwister(11), Float32, 4, 3)



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



        @test size(output.updated_vertices) == (4, 5)

        @test size(output.updated_hyperedges) == (3, 5)

        @test all(isfinite, output.updated_vertices)

        @test all(isfinite, output.updated_hyperedges)

    end



    @testset "Input validation without hyperedge features" begin

        layer = SpatialHypergraphLayer(4, 0, 8)

        ps, st = Lux.setup(MersenneTwister(12), layer)



        X_vertex = rand(Float32, 5, 4)

        H = ones(Float32, 5, 3)



        @test_throws ArgumentError layer(

            (X_vertex,),

            ps,

            st,

        )



        @test_throws ArgumentError layer(

            (

                X_vertex,

                rand(Float32, 3, 2),

                H,

            ),

            ps,

            st,

        )



        bad_vertex_features =

            rand(Float32, 5, 5)



        @test_throws DimensionMismatch layer(

            (bad_vertex_features, H),

            ps,

            st,

        )



        bad_incidence =

            ones(Float32, 4, 3)



        @test_throws DimensionMismatch layer(

            (X_vertex, bad_incidence),

            ps,

            st,

        )

    end



    @testset "Input validation with hyperedge features" begin

        layer = SpatialHypergraphLayer(4, 2, 8)

        ps, st = Lux.setup(MersenneTwister(13), layer)



        X_vertex = rand(Float32, 5, 4)

        H = ones(Float32, 5, 3)



        @test_throws ArgumentError layer(

            (X_vertex, H),

            ps,

            st,

        )



        wrong_hyperedge_count =

            rand(Float32, 4, 2)



        @test_throws DimensionMismatch layer(

            (

                X_vertex,

                wrong_hyperedge_count,

                H,

            ),

            ps,

            st,

        )



        wrong_hyperedge_features =

            rand(Float32, 3, 3)



        @test_throws DimensionMismatch layer(

            (

                X_vertex,

                wrong_hyperedge_features,

                H,

            ),

            ps,

            st,

        )

    end



    @testset "Deterministic initialization" begin

        layer = SpatialHypergraphLayer(4, 0, 8)



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

        @test ps_1.b_vertex == ps_2.b_vertex

        @test ps_1.W_hyperedge == ps_2.W_hyperedge

        @test ps_1.b_hyperedge == ps_2.b_hyperedge

        @test ps_1.W_vertex_update == ps_2.W_vertex_update

        @test ps_1.b_vertex_update == ps_2.b_vertex_update

    end

end

