from ml.population_models import employee_departure_probability, product_adoption_probability


def test_employee_model_responds_to_workplace_conditions():
    healthy = employee_departure_probability(1.2, .9, .1, 24, .2, 18, .9, .5)
    unhealthy = employee_departure_probability(.7, .2, .95, 2, -.2, 2, .2, .9)
    assert unhealthy > healthy


def test_product_model_responds_to_fit_and_usability():
    strong = product_adoption_probability(.9, .9, .9, .8, .2, 0, .6)
    weak = product_adoption_probability(.2, .2, .2, .1, .8, .4, .05)
    assert strong > weak
