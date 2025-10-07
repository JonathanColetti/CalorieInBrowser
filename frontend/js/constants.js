
const CLASSIFIER_MODEL_PATH = 'data/models/classifier/model.json';
const SEGMENTER_MODEL_PATH = 'data/models/segmenter/model.json';
const CLASSIFIER_IMG_SIZE = 168;
const SEGMENTATION_IMG_SIZE = 128;
const TOP_K_CLASSIFIER_RESULTS = 5;
const PIXELS_TO_GRAMS_FACTOR = 0.019; 

const CLASS_NAMES = [
    'Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare',
    'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 'Breakfast Burrito',
    'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 'Carrot Cake',
    'Ceviche', 'Cheesecake', 'Cheese Plate', 'Chicken Curry', 'Chicken Quesadilla',
    'Chicken Wings', 'Chocolate Cake', 'Chocolate Mousse', 'Churros', 'Clam Chowder',
    'Club Sandwich', 'Crab Cakes', 'Creme Brulee', 'Croque Madame', 'Cup Cakes',
    'Deviled Eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs Benedict',
    'Escargots', 'Falafel', 'Filet Mignon', 'Fish And Chips', 'Foie Gras',
    'French Fries', 'French Onion Soup', 'French Toast', 'Fried Calamari',
    'Fried Rice', 'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad',
    'Grilled Cheese Sandwich', 'Grilled Salmon', 'Guacamole', 'Gyoza',
    'Hamburger', 'Hot And Sour Soup', 'Hot Dog', 'Huevos Rancheros', 'Hummus',
    'Ice Cream', 'Lasagna', 'Lobster Bisque', 'Lobster Roll Sandwich',
    'Macaroni And Cheese', 'Macarons', 'Miso Soup', 'Mussels', 'Nachos',
    'Omelette', 'Onion Rings', 'Oysters', 'Pad Thai', 'Paella', 'Pancakes',
    'Panna Cotta', 'Peking Duck', 'Pho', 'Pizza', 'Pork Chop', 'Poutine',
    'Prime Rib', 'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake',
    'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed Salad',
    'Shrimp And Grits', 'Spaghetti Bolognese', 'Spaghetti Carbonara',
    'Spring Rolls', 'Steak', 'Strawberry Shortcake', 'Sushi', 'Tacos',
    'Takoyaki', 'Tiramisu', 'Tuna Tartare', 'Waffles'
];

const FOOD_DATABASE = {
    'Apple Pie': 237, 'Baby Back Ribs': 292, 'Baklava': 403, 'Beef Carpaccio': 129,
    'Beef Tartare': 146, 'Beet Salad': 65, 'Beignets': 363, 'Bibimbap': 125,
    'Bread Pudding': 152, 'Breakfast Burrito': 213, 'Bruschetta': 161, 'Caesar Salad': 158,
    'Cannoli': 274, 'Caprese Salad': 141, 'Carrot Cake': 408, 'Ceviche': 93, 'Cheesecake': 321,
    'Cheese Plate': 389, 'Chicken Curry': 137, 'Chicken Quesadilla': 282, 'Chicken Wings': 203,
    'Chocolate Cake': 371, 'Chocolate Mousse': 256, 'Churros': 447, 'Clam Chowder': 79,
    'Club Sandwich': 228, 'Crab Cakes': 201, 'Creme Brulee': 238, 'Croque Madame': 249,
    'Cup Cakes': 369, 'Deviled Eggs': 143, 'Donuts': 452, 'Dumplings': 225, 'Edamame': 122,
    'Eggs Benedict': 263, 'Escargots': 90, 'Falafel': 333, 'Filet Mignon': 267,
    'Fish And Chips': 195, 'Foie Gras': 462, 'French Fries': 312, 'French Onion Soup': 40,
    'French Toast': 229, 'Fried Calamari': 175, 'Fried Rice': 174, 'Frozen Yogurt': 159,
    'Garlic Bread': 350, 'Gnocchi': 156, 'Greek Salad': 99, 'Grilled Cheese Sandwich': 393,
    'Grilled Salmon': 208, 'Guacamole': 160, 'Gyoza': 184, 'Hamburger': 295,
    'Hot And Sour Soup': 41, 'Hot Dog': 290, 'Huevos Rancheros': 136, 'Hummus': 166,
    'Ice Cream': 207, 'Lasagna': 135, 'Lobster Bisque': 100, 'Lobster Roll Sandwich': 169,
    'Macaroni And Cheese': 376, 'Macarons': 384, 'Miso Soup': 33, 'Mussels': 172,
    'Nachos': 343, 'Omelette': 154, 'Onion Rings': 411, 'Oysters': 68, 'Pad Thai': 179,
    'Paella': 144, 'Pancakes': 227, 'Panna Cotta': 245, 'Peking Duck': 337, 'Pho': 70,
    'Pizza': 266, 'Pork Chop': 221, 'Poutine': 265, 'Prime Rib': 341,
    'Pulled Pork Sandwich': 189, 'Ramen': 138, 'Ravioli': 208, 'Red Velvet Cake': 407,
    'Risotto': 130, 'Samosa': 262, 'Sashimi': 124, 'Scallops': 111, 'Seaweed Salad': 120,
    'Shrimp And Grits': 121, 'Spaghetti Bolognese': 124, 'Spaghetti Carbonara': 352,
    'Spring Rolls': 246, 'Steak': 271, 'Strawberry Shortcake': 225, 'Sushi': 143,
    'Tacos': 226, 'Takoyaki': 98, 'Tiramisu': 240, 'Tuna Tartare': 156, 'Waffles': 291
};


const COLOR_MIN = 50;
const COLOR_MAX_RANDOM = 200;

const BACKGROUND_COLOR = [0, 0, 0];
const BACKGROUND_CLASS_ID = 0;
const SEGMENT_ALPHA = 180;
const SEGMENT_OVERLAY_OPACITY = 0.6;

const NORMALIZATION_DIVISOR = 255.0;
const BATCH_DIMENSION = 0;

const DEFAULT_CALORIES_PER_100G = 200;

const FULL_OPACITY = 1.0;

