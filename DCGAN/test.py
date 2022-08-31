from generator import Generator, get_noise
from discriminator import Discriminator


def test_gen():
    # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    '''
    Test your make_gen_block() function
    '''
    gen = Generator()
    num_test = 100

    # Test the hidden block
    test_hidden_noise = get_noise(num_test, gen.z_dim)
    test_hidden_block = gen.make_gen_block(10, 20, kernel_size=4, stride=1)
    test_uns_noise = gen.unsqueeze_noise(test_hidden_noise)
    hidden_output = test_hidden_block(test_uns_noise)

    # Check that it works with other strides
    test_hidden_block_stride = gen.make_gen_block(20, 20, kernel_size=4, stride=2)

    test_final_noise = get_noise(num_test, gen.z_dim) * 20
    test_final_block = gen.make_gen_block(10, 20, final_layer=True)
    test_final_uns_noise = gen.unsqueeze_noise(test_final_noise)
    final_output = test_final_block(test_final_uns_noise)

    # Test the whole thing:
    test_gen_noise = get_noise(num_test, gen.z_dim)
    test_uns_gen_noise = gen.unsqueeze_noise(test_gen_noise)
    gen_output = gen(test_uns_gen_noise)

    # UNIT TESTS
    assert tuple(hidden_output.shape) == (num_test, 20, 4, 4)
    assert hidden_output.max() > 1
    assert hidden_output.min() == 0
    assert hidden_output.std() > 0.2
    assert hidden_output.std() < 1
    assert hidden_output.std() > 0.5

    assert tuple(test_hidden_block_stride(hidden_output).shape) == (num_test, 20, 10, 10)

    assert final_output.max().item() == 1
    assert final_output.min().item() == -1

    assert tuple(gen_output.shape) == (num_test, 1, 28, 28)
    assert gen_output.std() > 0.5
    assert gen_output.std() < 0.8
    print("Success!")


def test_dis():
    # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    '''
    Test your make_disc_block() function
    '''
    num_test = 100

    gen = Generator()
    disc = Discriminator()
    test_images = gen(get_noise(num_test, gen.z_dim))

    # Test the hidden block
    test_hidden_block = disc.make_disc_block(1, 5, kernel_size=6, stride=3)
    hidden_output = test_hidden_block(test_images)

    # Test the final block
    test_final_block = disc.make_disc_block(1, 10, kernel_size=2, stride=5, final_layer=True)
    final_output = test_final_block(test_images)

    # Test the whole thing:
    disc_output = disc(test_images)

    # Test the hidden block
    assert tuple(hidden_output.shape) == (num_test, 5, 8, 8)
    # Because of the LeakyReLU slope
    assert -hidden_output.min() / hidden_output.max() > 0.15
    assert -hidden_output.min() / hidden_output.max() < 0.25
    assert hidden_output.std() > 0.5
    assert hidden_output.std() < 1

    # Test the final block

    assert tuple(final_output.shape) == (num_test, 10, 6, 6)
    assert final_output.max() > 1.0
    assert final_output.min() < -1.0
    assert final_output.std() > 0.3
    assert final_output.std() < 0.6

    # Test the whole thing:

    assert tuple(disc_output.shape) == (num_test, 1)
    assert disc_output.std() > 0.25
    assert disc_output.std() < 0.5
    print("Success!")


