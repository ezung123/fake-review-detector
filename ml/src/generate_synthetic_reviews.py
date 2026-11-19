import random
import pandas as pd
from pathlib import Path

random.seed(42)

# Specific hotel names across various cities
HOTEL_NAMES = [
    "The Langham Chicago", "Grand Hyatt Seattle", "Kimpton Hotel Monaco Denver",
    "Fairmont Austin", "The Ritz-Carlton San Francisco", "Omni Shoreham Washington DC",
    "Marriott Marquis Atlanta", "The Westin Copley Place Boston", "Hilton Midtown New York",
    "JW Marriott Miami", "The Roosevelt New Orleans", "Loews Philadelphia Hotel",
    "Hyatt Regency Dallas", "InterContinental Minneapolis", "Sheraton Grand Phoenix",
    "The Drake Chicago", "Hotel Emma San Antonio", "The Nines Portland",
    "W Nashville", "Sofitel Philadelphia at Rittenhouse Square"
]

ROOM_TYPES = [
    "corner king suite", "deluxe queen room", "executive skyline room",
    "standard double room", "ocean-view king room", "junior suite",
    "premier high-floor room", "standard king room", "terrace suite"
]

STAFF_ROLES = [
    "front desk receptionist", "concierge team", "valet attendant",
    "night auditor", "housekeeping supervisor", "bellhop", "guest services manager"
]

# Positive components
POS_OPENINGS = [
    "I recently spent three nights at {hotel} for a {trip_type}, and the overall experience was exceptional.",
    "From the moment we pulled up to {hotel}, we were thoroughly impressed by the welcoming atmosphere.",
    "Our stay at {hotel} in a {room} exceeded our expectations in virtually every way.",
    "If you're visiting the city, {hotel} is an outstanding choice that offers both comfort and convenience.",
    "We booked {hotel} based on a friend's recommendation, and it certainly lived up to the praise.",
    "My partner and I spent a weekend getaway at {hotel}, and it was truly memorable from start to finish.",
    "Having stayed at numerous business hotels throughout the country, {hotel} stands out for its impeccable service.",
    "I had a fantastic four-night stay at {hotel} during an annual corporate conference.",
    "Choosing {hotel} was the best decision we made for our family vacation this season.",
    "From check-in to check-out, our experience at {hotel} was virtually flawless."
]

POS_ROOM_DETAILS = [
    "The {room} was spotless, spacious, and featured floor-to-ceiling windows with panoramic city views.",
    "The plush king mattress and premium linens provided some of the most restful sleep I've had while traveling.",
    "Housekeeping kept the room immaculate, and the marble bathroom with a rainfall shower was a luxurious touch.",
    "I particularly appreciated the thoughtful modern amenities, including fast Wi-Fi, ample USB charging ports, and a Nespresso machine.",
    "The blackout shades worked perfectly, ensuring complete silence and darkness despite being in the heart of downtown.",
    "The room was impeccably maintained with stylish contemporary decor, quiet climate control, and generous closet space."
]

POS_AMENITY_DETAILS = [
    "The morning breakfast buffet offered an impressive array of fresh pastries, made-to-order omelets, and artisanal coffee.",
    "We made great use of the rooftop fitness center, which was well-equipped with modern cardio machines and clean towels.",
    "The hotel lounge on the top floor had superb craft cocktails and light bites with breathtaking skyline views.",
    "Valet parking was quick and efficient, with our car retrieved in under five minutes every time.",
    "The location is unbeatable, situated within comfortable walking distance to top museums, boutiques, and transit stations.",
    "Room service arrived promptly within 25 minutes, hot and beautifully presented."
]

POS_STAFF_DETAILS = [
    "The {staff} greeted us warmly by name and provided fantastic recommendations for local dining spots off the tourist path.",
    "Every member of the staff we encountered went out of their way to ensure our comfort with genuine professionalism.",
    "When our incoming flight was delayed, the front desk handled our late arrival seamlessly without any hiccups.",
    "The concierge was extraordinarily helpful in securing last-minute theater tickets and dinner reservations on a busy Saturday night."
]

POS_CLOSINGS = [
    "All in all, I would return to {hotel} without hesitation and recommend it enthusiastically to anyone visiting the area.",
    "It is rare to find a property that balances luxury, location, and genuine hospitality so well. Five stars all around.",
    "We are already planning our next stay here. Thank you to the entire team for a wonderful and rejuvenating trip!",
    "A wonderful hospitality experience that justified every penny. I will definitely be back on my next business trip.",
    "Highly recommended for both leisure travelers and business professionals seeking a dependable, high-quality stay."
]

# Negative components
NEG_OPENINGS = [
    "Unfortunately, our recent stay at {hotel} fell remarkably short of what we anticipated for the price.",
    "I rarely write critical reviews, but my experience at {hotel} this past week was deeply disappointing.",
    "We booked a {room} at {hotel} hoping for a peaceful weekend retreat, but encountered continuous issues from day one.",
    "For a property that markets itself as an upscale establishment, {hotel} delivers lackluster service and neglected facilities.",
    "I had high expectations for {hotel}, but our four-night business stay was fraught with frustration and inconvenience.",
    "Save your money and look elsewhere; {hotel} does not deliver the standard of hospitality you would expect.",
    "Our stay at {hotel} was marred by poor communication, noise disturbances, and an indifferent front desk.",
    "I checked into {hotel} for a three-day conference, and the series of mishaps began before I even reached my room."
]

NEG_ROOM_DETAILS = [
    "The {room} felt dated and neglected; carpet stains were visible near the entrance, and the wallpaper was peeling behind the headboard.",
    "The air conditioning unit was excessively loud, constantly rattling throughout the night and failing to keep the room cool.",
    "Water pressure in the shower was barely a trickle, and it took over five minutes for the water to become lukewarm.",
    "We specifically requested a quiet room on a high floor, but were placed right next to the elevators with constant dinging and foot traffic.",
    "The bedding felt scratchy and thin, and the pillows were lumpy and lacked proper support.",
    "The room smelled faintly of stale smoke and dampness, despite being designated as a strictly non-smoking property."
]

NEG_AMENITY_DETAILS = [
    "The advertised fitness center had two out of three treadmills out of order and no clean towels available.",
    "Wi-Fi was painfully slow and disconnected repeatedly, making it nearly impossible to conduct remote work meetings.",
    "We waited over forty-five minutes for valet parking to bring our car around, almost causing us to miss our dinner reservation.",
    "The breakfast was overpriced and mediocre, featuring lukewarm eggs, stale bread, and burnt drip coffee.",
    "There was ongoing construction noise in the adjacent wing starting promptly at 6:30 AM with zero advance warning from management.",
    "We were surprised by unexpected 'urban destination' resort fees of $45 per day that were never clearly explained at booking."
]

NEG_STAFF_DETAILS = [
    "When I brought these concerns to the {staff}, I was met with an unhelpful attitude and a dismissive shrug.",
    "Check-in took over thirty-five minutes due to an understaffed front desk, and no apology was offered for the delay.",
    "Housekeeping completely missed servicing our room on the second day, despite having the service sign displayed.",
    "Management seemed completely indifferent to guest feedback, promising a follow-up call from a manager that never arrived."
]

NEG_CLOSINGS = [
    "Overall, this was a thoroughly frustrating experience, and I will not be returning to {hotel} in the future.",
    "Given the exorbitant nightly rate and the sheer number of quality hotels nearby, I strongly advise staying elsewhere.",
    "I expected much higher standards from this brand. A very disappointing stay that definitely ruined our weekend.",
    "Until management addresses basic maintenance and staff training, I cannot recommend {hotel} to anyone.",
    "There are far better accommodations in the city that offer genuine value for your money. Proceed with caution."
]

TRIP_TYPES = ["weekend anniversary trip", "business trip", "family vacation", "leisure getaway", "solo conference trip"]

def generate_positive_review(idx: int) -> str:
    hotel = HOTEL_NAMES[idx % len(HOTEL_NAMES)]
    room = ROOM_TYPES[idx % len(ROOM_TYPES)]
    staff = STAFF_ROLES[idx % len(STAFF_ROLES)]
    trip = TRIP_TYPES[idx % len(TRIP_TYPES)]
    
    opening = POS_OPENINGS[idx % len(POS_OPENINGS)].format(hotel=hotel, room=room, trip_type=trip)
    room_detail = POS_ROOM_DETAILS[idx % len(POS_ROOM_DETAILS)].format(room=room)
    amenity_detail = POS_AMENITY_DETAILS[idx % len(POS_AMENITY_DETAILS)]
    staff_detail = POS_STAFF_DETAILS[idx % len(POS_STAFF_DETAILS)].format(staff=staff)
    closing = POS_CLOSINGS[idx % len(POS_CLOSINGS)].format(hotel=hotel)
    
    # Vary the length: some shorter (3 sentences), some longer (all sentences)
    pattern = idx % 3
    if pattern == 0:
        return f"{opening} {room_detail} {closing}"
    elif pattern == 1:
        return f"{opening} {room_detail} {amenity_detail} {closing}"
    else:
        return f"{opening} {room_detail} {amenity_detail} {staff_detail} {closing}"

def generate_negative_review(idx: int) -> str:
    hotel = HOTEL_NAMES[(idx + 7) % len(HOTEL_NAMES)]
    room = ROOM_TYPES[(idx + 4) % len(ROOM_TYPES)]
    staff = STAFF_ROLES[(idx + 3) % len(STAFF_ROLES)]
    
    opening = NEG_OPENINGS[idx % len(NEG_OPENINGS)].format(hotel=hotel, room=room)
    room_detail = NEG_ROOM_DETAILS[idx % len(NEG_ROOM_DETAILS)].format(room=room)
    amenity_detail = NEG_AMENITY_DETAILS[idx % len(NEG_AMENITY_DETAILS)]
    staff_detail = NEG_STAFF_DETAILS[idx % len(NEG_STAFF_DETAILS)].format(staff=staff)
    closing = NEG_CLOSINGS[idx % len(NEG_CLOSINGS)].format(hotel=hotel)
    
    pattern = idx % 3
    if pattern == 0:
        return f"{opening} {room_detail} {closing}"
    elif pattern == 1:
        return f"{opening} {room_detail} {amenity_detail} {closing}"
    else:
        return f"{opening} {room_detail} {amenity_detail} {staff_detail} {closing}"

def main():
    rows = []
    
    # 100 positive reviews
    for i in range(100):
        text = generate_positive_review(i)
        rows.append({
            "row_id": f"synthetic_hotel_ai_pos_{i+1:03d}",
            "text": text,
            "sentiment": "positive",
            "source_dataset": "synthetic_hotel_ai",
            "authorship_label": "ai",
            "is_synthetic": True
        })
        
    # 100 negative reviews
    for i in range(100):
        text = generate_negative_review(i)
        rows.append({
            "row_id": f"synthetic_hotel_ai_neg_{i+1:03d}",
            "text": text,
            "sentiment": "negative",
            "source_dataset": "synthetic_hotel_ai",
            "authorship_label": "ai",
            "is_synthetic": True
        })
        
    df = pd.DataFrame(rows)
    out_path = Path("data/processed/synthetic_hotel_ai.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic AI reviews saved to {out_path.resolve()}")
    print("Class distribution:")
    print(df["sentiment"].value_counts())
    print("\nSample Positive Review:")
    print(df.iloc[0]["text"])
    print("\nSample Negative Review:")
    print(df.iloc[100]["text"])

if __name__ == "__main__":
    main()
