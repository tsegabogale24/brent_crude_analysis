from flask import Blueprint, jsonify
import pandas as pd

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/prices', methods=['GET'])
def get_prices():
    df = pd.read_csv('../data/BrentOilPrices.csv', parse_dates=['Date'])
    return df.to_json(orient='records', date_format='iso')

@api_bp.route('/api/events', methods=['GET'])
def get_events():
    df = pd.read_csv('data/event_data.csv', parse_dates=['Date'])
    return df.to_json(orient='records', date_format='iso')

@api_bp.route('/api/change-points', methods=['GET'])
def get_change_points():
    df = pd.read_csv('data/change_points.csv', parse_dates=['Date'])
    return df.to_json(orient='records', date_format='iso')
