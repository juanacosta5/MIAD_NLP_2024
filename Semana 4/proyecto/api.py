from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
from flask_cors import CORS
import requests
from model_deployment import predict
app = Flask(__name__)
CORS(app)  
api = Api(
    app, 
    version='1.0', 
    title='Predictor de Precios de Autos',
    description='API para predecir precios de autos')

ns = api.namespace('predict', 
     description='Predictor de Precio')

parser = reqparse.RequestParser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Año del automóvil', 
    location='args')
parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Millaje del automóvil', 
    location='args')
parser.add_argument(
    'State', 
    type=int, 
    required=True, 
    help='Estado en el que se encuentra del automóvil', 
    location='args')
parser.add_argument(
    'Make', 
    type=int, 
    required=True, 
    help='Marca del automóvil', 
    location='args')
parser.add_argument(
    'Model', 
    type=int, 
    required=True, 
    help='Modelo del automóvil', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/cost-prediction')
class CarPricePrediction(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        year = args['Year']
        mileage = args['Mileage']
        state = args['State']
        make = args['Make']
        model = args['Model']
        print(year)
        print(mileage)
        print(state)
        print(make)
        print(model)

        # prediction_service_url = "http://localhost:5000/predict"
        data = {
            'Year': year,
            'Mileage': mileage,
            'State': state,
            'Make': make,
            'Model': model
        }
        response = predict(data)
        print (response)
        
        if response[1] == 200:
            result = response[0]
            return {'result': result}, 200
        else:
            return {'error': 'No se pudo realizar la predicción'}, 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
