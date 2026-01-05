"""Generate 2000 Puerto Rico addresses with urbanization data."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# PR Urbanizations (real ones)
URBANIZATIONS = [
    'Villa Carolina', 'Las Lomas', 'Country Club', 'Villa Nevarez', 'Caparra Terrace',
    'University Gardens', 'Puerto Nuevo', 'Levittown', 'Torrimar', 'Garden Hills',
    'Altamira', 'Caparra Heights', 'El Comandante', 'Floral Park', 'Hyde Park',
    'Jardines de Caparra', 'Los Paseos', 'Mansiones de Rio Piedras', 'Monte Park',
    'Parque de las Fuentes', 'Quintana', 'Reparto Metropolitano', 'Roosevelt',
    'San Patricio', 'Santa Maria', 'Sierra Bayamon', 'Summit Hills', 'Urb Mariolga',
    'Venus Gardens', 'Villa Capri', 'Villa del Rey', 'Villa Espana', 'Villas de Castro',
    'Vistas del Turabo', 'Estancias del Cafetal', 'Paseo Las Olas', 'Cond El Monte',
    'Ext San Agustin', 'El Paraiso', 'La Riviera', 'Los Maestros',
    'Santa Rosa', 'Vista Hermosa', 'El Verde', 'Bonneville Heights',
    'Bairoa', 'Caguas Norte', 'Lomas Verdes', 'Munoz Rivera'
]

# PR Cities with ZIP ranges
PR_CITIES = [
    ('San Juan', ['00901', '00902', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00928', '00929', '00931', '00933', '00934', '00936', '00940']),
    ('Carolina', ['00979', '00981', '00982', '00983', '00984', '00985', '00987', '00988']),
    ('Bayamon', ['00956', '00957', '00958', '00959', '00960', '00961']),
    ('Ponce', ['00716', '00717', '00728', '00730', '00731', '00732', '00733', '00734']),
    ('Caguas', ['00725', '00726', '00727']),
    ('Mayaguez', ['00680', '00681', '00682']),
    ('Guaynabo', ['00965', '00966', '00968', '00969', '00970', '00971']),
    ('Trujillo Alto', ['00976', '00977']),
    ('Arecibo', ['00612', '00613', '00614']),
    ('Fajardo', ['00738', '00740']),
    ('Humacao', ['00791', '00792']),
    ('Aguadilla', ['00603', '00604', '00605']),
    ('Manati', ['00674']),
    ('Vega Baja', ['00693', '00694']),
    ('Rio Grande', ['00745']),
    ('Dorado', ['00646']),
    ('Toa Baja', ['00949', '00950', '00951']),
    ('Toa Alta', ['00953', '00954']),
    ('Catano', ['00962', '00963']),
    ('Loiza', ['00772'])
]

# Spanish street names
STREET_NAMES = [
    'Calle A', 'Calle B', 'Calle C', 'Calle D', 'Calle E', 'Calle F', 'Calle G',
    'Calle 1', 'Calle 2', 'Calle 3', 'Calle 4', 'Calle 5', 'Calle 6', 'Calle 7',
    'Avenida Central', 'Avenida Principal', 'Calle del Sol', 'Calle Luna',
    'Calle Estrella', 'Calle Rosa', 'Calle Lirio', 'Calle Margarita',
    'Calle Gardenia', 'Calle Jazmin', 'Calle Norte', 'Calle Sur',
    'Calle Este', 'Calle Oeste', 'Calle Primera', 'Calle Segunda',
    'Calle Tercera', 'Calle Cuarta', 'Calle Quinta', 'Ave Hostos',
    'Ave Munoz Rivera', 'Ave Ponce de Leon', 'Ave Fernandez Juncos',
    'Ave Roosevelt', 'Ave Kennedy', 'Calle San Jorge', 'Calle San Jose',
    'Calle Loiza', 'Calle McLeary', 'Ave Ashford', 'Calle del Parque'
]

# PR first/last names
FIRST_NAMES = [
    'Maria', 'Jose', 'Juan', 'Carmen', 'Luis', 'Ana', 'Carlos', 'Rosa', 'Miguel', 'Isabel',
    'Pedro', 'Elena', 'Francisco', 'Teresa', 'Antonio', 'Marta', 'Rafael', 'Gloria', 'Manuel', 'Luz',
    'Jorge', 'Cristina', 'Roberto', 'Patricia', 'David', 'Beatriz', 'Hector', 'Laura', 'Angel', 'Diana',
    'Fernando', 'Sandra', 'Javier', 'Silvia', 'Ricardo', 'Monica', 'Eduardo', 'Adriana', 'Alejandro', 'Paula',
    'Enrique', 'Carolina', 'Andres', 'Daniela', 'Oscar', 'Veronica', 'Raul', 'Gabriela', 'Sergio', 'Natalia'
]

LAST_NAMES = [
    'Rodriguez', 'Martinez', 'Garcia', 'Lopez', 'Gonzalez', 'Rivera', 'Torres', 'Ramirez', 'Cruz', 'Ortiz',
    'Morales', 'Reyes', 'Santiago', 'Colon', 'Diaz', 'Hernandez', 'Ruiz', 'Perez', 'Sanchez', 'Vargas',
    'Ramos', 'Rosario', 'Figueroa', 'Acosta', 'Medina', 'Vega', 'Mendez', 'Castro', 'Delgado', 'Soto',
    'Nieves', 'Maldonado', 'Miranda', 'Rojas', 'Pagan', 'Molina', 'Padilla', 'Cordero', 'Ayala', 'Serrano',
    'Alicea', 'Baez', 'Burgos', 'Caraballo', 'Cintron', 'Correa', 'Feliciano', 'Franco', 'Guzman', 'Leon'
]


def random_date(start_year=2020, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')


def generate_pr_customer(index):
    city, zips = random.choice(PR_CITIES)
    zip_code = random.choice(zips)
    urb = random.choice(URBANIZATIONS)
    street = random.choice(STREET_NAMES)
    street_num = random.randint(1, 999)

    # Some addresses have apt/unit
    apt = ''
    if random.random() < 0.3:
        apt = f' Apt {random.randint(1, 500)}'

    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)

    # Build address - some with URB prefix, some with URBANIZACION
    urb_prefix = random.choice(['URB ', 'Urb ', 'URBANIZACION ', 'Urbanizacion '])
    address = f'{urb_prefix}{urb}, {street_num} {street}{apt}'

    move_count = random.randint(0, 5)
    created = random_date(2018, 2023)
    last_move = random_date(2023, 2025) if move_count > 0 else None

    # Generate move history
    move_history = []
    for i in range(move_count):
        prev_city, prev_zips = random.choice(PR_CITIES)
        move_history.append({
            'from_address': None if i == 0 else f'{random.randint(1,999)} {random.choice(STREET_NAMES)}',
            'to_address': f'{random.randint(1,999)} {random.choice(STREET_NAMES)}',
            'city': prev_city,
            'state': 'PR',
            'zip': random.choice(prev_zips),
            'move_date': random_date(2020, 2025)
        })

    return {
        'crid': f'CRID-PR{str(index).zfill(4)}',
        'name': f'{first} {last}',
        'first_name': first,
        'last_name': last,
        'address': address,
        'city': city,
        'state': 'PR',
        'zip': zip_code,
        'customer_type': random.choice(['RESIDENTIAL', 'RESIDENTIAL', 'RESIDENTIAL', 'BUSINESS', 'PO_BOX']),
        'status': random.choice(['ACTIVE', 'ACTIVE', 'ACTIVE', 'INACTIVE', 'PENDING']),
        'move_count': move_count,
        'last_move': last_move,
        'created_date': created,
        'move_history': move_history
    }


def main():
    # Generate 2000 PR customers
    print("Generating 2000 PR addresses...")
    pr_customers = [generate_pr_customer(i + 1) for i in range(2000)]

    # Read existing data
    data_file = Path(__file__).parent.parent / 'customer_data.json'
    print(f"Reading existing data from {data_file}")

    with open(data_file, 'r') as f:
        existing = json.load(f)

    original_count = len(existing)
    print(f"Original customer count: {original_count}")

    # Append PR customers
    existing.extend(pr_customers)

    # Write back
    with open(data_file, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"Added {len(pr_customers)} PR addresses")
    print(f"Total customers now: {len(existing)}")

    # Show some samples
    print("\nSample PR addresses:")
    for i in range(5):
        c = pr_customers[i]
        print(f"  {c['crid']}: {c['address']}, {c['city']}, PR {c['zip']}")


if __name__ == '__main__':
    main()
